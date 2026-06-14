"""Titans Long-term Memory Module (LMM).

Implements an associative memory matrix M_t with momentum S_t and
data-dependent gating (forgetting, momentum, learning rate). The loss is
self-supervised: || M(k_t) - v_t ||^2.

CRITICAL RULES:
- Key and value are projections of the SAME hidden vector.
- M and S are PERSISTENT across sessions.
- Input must be DETACHED (gradient barrier from TTT).
- W_key, W_val and the gating networks are FIXED random initializations.
  Only M and S are updated online via the Titans rule. This removes the
  ambiguity of "who updates what" and keeps the online learning signal
  focused on the associative memory matrix.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoesisTitansLMM:
    def __init__(self, dim=1024, device="cpu"):
        self.dim = dim
        self.device = device

        # Persistent memory state (CPU/RAM by default for 8GB systems).
        self.M = torch.zeros(dim, dim, device=device)
        self.S = torch.zeros(dim, dim, device=device)

        # Key-Value projections: learned via Titans rule, not standard backprop.
        self.W_key = nn.Linear(dim, dim, bias=False).to(device)
        self.W_val = nn.Linear(dim, dim, bias=False).to(device)

        # Data-dependent gating networks per token.
        self.gate_forget = nn.Sequential(
            nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        ).to(device)
        self.gate_momentum = nn.Sequential(
            nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        ).to(device)
        self.gate_lr = nn.Sequential(
            nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid()
        ).to(device)

        # Base scaling factors (tuned per domain; start conservative).
        self.alpha_base = 0.05
        self.eta_base = 0.90
        self.theta_base = 0.001

        # Freeze key/value projections and gating networks. They are treated as
        # fixed random transforms; online learning happens only in M and S.
        for p in self.W_key.parameters():
            p.requires_grad = False
        for p in self.W_val.parameters():
            p.requires_grad = False
        for gate in (self.gate_forget, self.gate_momentum, self.gate_lr):
            for p in gate.parameters():
                p.requires_grad = False

    def learn(self, hidden_states):
        """
        hidden_states: (seq_len, dim) — DETACHED TTT output, on CPU.
        Performs one Titans update step per token.
        Returns final reconstruction loss scalar.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = hidden_states.to(self.device)

        # Project to key and value spaces.
        k = self.W_key(hidden_states)  # (seq, dim)
        v = self.W_val(hidden_states)  # (seq, dim)

        final_loss = 0.0
        for t in range(len(hidden_states)):
            k_t = k[t]
            v_t = v[t]

            # Memory read: reconstruct v from k using M.
            v_pred = self.M @ k_t

            # Self-supervised reconstruction loss.
            loss = F.mse_loss(v_pred, v_t)

            # Manual gradient of loss w.r.t. M.
            grad = torch.outer(v_pred - v_t, k_t)

            # Data-dependent gating per token.
            token_emb = hidden_states[t]
            alpha = self.gate_forget(token_emb).item() * self.alpha_base
            eta = self.gate_momentum(token_emb).item() * self.eta_base
            theta = self.gate_lr(token_emb).item() * self.theta_base

            # Titans momentum update.
            self.S = eta * self.S - theta * grad

            # Memory update with forgetting.
            self.M = (1 - alpha) * self.M + self.S

            final_loss = loss.item()

        return final_loss

    def retrieve(self, query_hidden):
        """
        query_hidden: (dim,) on CPU.
        Returns: (retrieved_value, confidence_score)
        """
        if query_hidden.dim() == 2:
            query_hidden = query_hidden.squeeze(0)
        query_hidden = query_hidden.to(self.device)

        with torch.no_grad():
            k_q = self.W_key(query_hidden)
            v_r = self.M @ k_q

            # Confidence: how well does v_r reconstruct back to k space?
            k_reconstructed = self.W_key(v_r)
            confidence = F.cosine_similarity(
                k_q.unsqueeze(0), k_reconstructed.unsqueeze(0)
            ).item()
            confidence = max(-1.0, min(1.0, confidence))

        return v_r, confidence

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "M": self.M,
                "S": self.S,
                "W_key": self.W_key.state_dict(),
                "W_val": self.W_val.state_dict(),
                "gates": [
                    self.gate_forget.state_dict(),
                    self.gate_momentum.state_dict(),
                    self.gate_lr.state_dict(),
                ],
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.M = state["M"].to(self.device)
        self.S = state["S"].to(self.device)
        self.W_key.load_state_dict(state["W_key"])
        self.W_val.load_state_dict(state["W_val"])
        gates = [self.gate_forget, self.gate_momentum, self.gate_lr]
        for g, sd in zip(gates, state["gates"]):
            g.load_state_dict(sd)
