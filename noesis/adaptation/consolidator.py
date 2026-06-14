"""Nightly/background consolidation pipeline.

When enough interactions are accumulated, train a small domain-specific
logit-head adapter on the highest-surprise traces and add it to the MoE as a
new expert.

The adapter learns to map the TTT-adapted hidden state to the actual token
distribution the model observed. This is a language-modeling objective over
the raw interaction tokens, not an autoencoder: the new expert must predict
which tokens appeared, not reconstruct its own input.

Optionally, EWC (Elastic Weight Consolidation) regularization penalizes large
deviations from the currently active experts, reducing catastrophic forgetting
when a new expert replaces an old slot.
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

    def run(
        self,
        topk=500,
        min_traces=100,
        epochs=5,
        val_split=0.1,
        use_ewc=True,
        ewc_lambda=100.0,
        fisher_samples=200,
    ):
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

        # EWC setup: remember old parameters and Fisher diagonal of active experts.
        fisher_reg = {}
        old_params = {}
        if use_ewc:
            fisher_reg = self._compute_fisher(
                self.engine.moe, X_train, Y_train, n_samples=fisher_samples
            )
            old_params = {
                name: param.clone().detach()
                for name, param in adapter.named_parameters()
            }

        best_val_loss = float("inf")
        for epoch in range(epochs):
            adapter.train()
            pred = adapter(X_train)  # (N, vocab_size)
            loss = F.cross_entropy(pred, Y_train)

            if use_ewc and fisher_reg and old_params:
                ewc_loss = 0.0
                for name, param in adapter.named_parameters():
                    if name in fisher_reg and name in old_params:
                        ewc_loss += (
                            fisher_reg[name] * (param - old_params[name]) ** 2
                        ).sum()
                loss = loss + ewc_lambda * ewc_loss

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

        # Activate in MoE. If no empty slot, back up the least-used expert first.
        moe = self.engine.moe
        empty = (moe.expert_mask == 0).nonzero(as_tuple=False)
        if len(empty) > 0:
            target_slot = empty[0].item()
        else:
            target_slot = self._find_least_used_slot(moe)
            backup_path = (
                f"noesis_experts/backup_slot{target_slot}_{int(time.time())}.pt"
            )
            existing_state = {
                k: v.clone() for k, v in moe.active_experts[target_slot].state_dict().items()
            }
            torch.save(existing_state, backup_path)
            print(f"Backed up existing expert in slot {target_slot} to {backup_path}")

        moe.swap_expert(target_slot, path)
        if hasattr(moe, "expert_usage"):
            moe.expert_usage[target_slot] = 0.0

        # Clear traces.
        self.engine.session_traces = []
        return path

    def _find_least_used_slot(self, moe):
        """Return the active expert slot with the lowest usage count.

        Falls back to slot 0 if no usage statistics have been collected yet.
        """
        if not hasattr(moe, "expert_usage"):
            return 0
        return torch.argmin(moe.expert_usage).item()

    def _compute_fisher(self, moe, X, Y, n_samples=200):
        """Estimate the diagonal Fisher information of the active experts.

        The Fisher values are averaged across all active experts and normalized
        by the number of samples used. The resulting keys match the parameter
        names of a standard MoE expert / consolidation adapter, so they can be
        used directly for EWC regularization.
        """
        fisher = {}
        n = min(n_samples, len(X))
        if n == 0:
            return fisher
        idx = torch.randperm(len(X))[:n]

        total_samples = 0
        for expert in moe.active_experts:
            expert.train()
            for i in idx:
                expert.zero_grad()
                out = expert(X[i].unsqueeze(0))
                loss = F.cross_entropy(out, Y[i].unsqueeze(0))
                loss.backward()
                for name, param in expert.named_parameters():
                    if param.grad is not None:
                        fisher[name] = fisher.get(name, 0.0) + param.grad.data ** 2
                total_samples += 1
            expert.eval()

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        return fisher
