"""Nightly/background consolidation pipeline.

When enough interactions are accumulated, train a small domain-specific
logit-head adapter on the highest-surprise traces and add it to the MoE as a
new expert.

The adapter learns to map the TTT-adapted hidden state to the actual token
distribution the model observed. This is a language-modeling objective over
the raw interaction tokens, not an autoencoder: the new expert must predict
which tokens appeared, not reconstruct its own input.
"""

import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoesisConsolidator:
    def __init__(self, engine):
        self.engine = engine

    def run(self, topk=500, min_traces=100, epochs=5, val_split=0.1):
        """Train a domain logit adapter on high-surprise traces and install it as an expert."""
        traces = sorted(
            self.engine.session_traces,
            key=lambda x: x.get("surprise", 0.0),
            reverse=True,
        )[:topk]

        if len(traces) < min_traces:
            warnings.warn(
                f"Consolidation requires at least {min_traces} traces; got {len(traces)}."
            )
            return None

        d_model = self.engine.ttt.d_model
        vocab_size = self.engine.moe.vocab_size

        # Build dataset: X = TTT-adapted hidden states, Y = observed tokens.
        X, Y = [], []
        for tr in traces:
            x = tr.get("ttt_out")
            y = tr.get("input_ids")
            if x is None or y is None:
                continue
            if x.dim() == 3:
                x = x.squeeze(0)
            if y.dim() == 2:
                y = y.squeeze(0)
            if x.shape[0] != y.shape[0]:
                continue
            if x.shape[-1] != d_model:
                continue
            X.append(x)
            Y.append(y)

        if len(X) < min_traces:
            warnings.warn(
                f"Not enough valid (ttt_out, input_ids) pairs; got {len(X)}."
            )
            return None

        X = torch.cat(X, dim=0)  # (total_tokens, d_model)
        Y = torch.cat(Y, dim=0)  # (total_tokens,)

        # Clamp token ids to the vocabulary the MoE knows about.
        Y = Y.clamp(0, vocab_size - 1)

        # Train/validation split.
        n = len(X)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        perm = torch.randperm(n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # Domain-specific logit adapter. Matches the MoE expert architecture so
        # it can be hot-swapped into an active expert slot.
        adapter = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, vocab_size),
        )
        opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            adapter.train()
            pred = adapter(X_train)  # (N, vocab_size)
            loss = F.cross_entropy(pred, Y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(X_val)
                val_loss = F.cross_entropy(val_pred, Y_val).item()

            print(
                f"Consolidation epoch {epoch + 1}/{epochs}: "
                f"train_loss={loss.item():.6f} val_loss={val_loss:.6f}"
            )
            best_val_loss = min(best_val_loss, val_loss)

        # Save to disk.
        os.makedirs("noesis_experts", exist_ok=True)
        path = f"noesis_experts/domain_{int(time.time())}.pt"
        torch.save(adapter.state_dict(), path)

        # Activate in MoE.
        moe = self.engine.moe
        empty = (moe.expert_mask == 0).nonzero(as_tuple=False)
        if len(empty) > 0:
            moe.swap_expert(empty[0].item(), path)
        else:
            # Replace lowest-usage active expert (LRU heuristic: slot 0).
            moe.swap_expert(0, path)

        # Clear traces.
        self.engine.session_traces = []
        return path
