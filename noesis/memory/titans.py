"""Titans Long-term Memory Module (LMM).

Implements an associative memory matrix M_t with momentum S_t and
data-dependent gating (forgetting, momentum, learning rate).

CRITICAL RULES:
- Key and value are projections of the SAME hidden vector.
- M and S are PERSISTENT across sessions and updated by the Titans rule.
- Input must be DETACHED (gradient barrier from TTT).
- W_key and W_val are LEARNED end-to-end from the reconstruction loss.
- Gating networks are LEARNED via a surrogate loss: the gate outputs are
  scored by whether the resulting M update improved retrieval quality for
  the current token. M and S themselves remain persistent and updated
  outside the autograd graph for efficiency.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoesisTitansLMM:
    def __init__(self, dim=1024, device="cpu", gate_lr=1e-4):
        self.dim = dim
        self.device = device

        # Persistent memory state (CPU/RAM by default for 8GB systems).
        self.M = torch.zeros(dim, dim, device=device)
        self.S = torch.zeros(dim, dim, device=device)

        # Key-Value projections: learned via the reconstruction loss.
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

        # Joint optimizer for key/value projections and gating networks.
        self.gate_optimizer = torch.optim.Adam(
            list(self.gate_forget.parameters())
            + list(self.gate_momentum.parameters())
            + list(self.gate_lr.parameters())
            + list(self.W_key.parameters())
            + list(self.W_val.parameters()),
            lr=gate_lr,
        )

        # Online token counter for periodic capacity management.
        self._learn_count = 0

    def learn(self, hidden_states, surprise=None):
        """
        hidden_states: (seq_len, dim) — DETACHED TTT output, on CPU.
        surprise: optional scalar surprise signal from the TTT layer.

        Performs one Titans update step per token and periodically updates the
        key/value projections and gating networks. Gates are trained by a
        surrogate loss that measures whether the M update improved retrieval
        quality for the current token.
        Returns final reconstruction loss scalar.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = hidden_states.to(self.device)

        gate_period = 16
        accumulated_loss = 0.0
        accumulated_steps = 0
        final_loss = 0.0
        gate_loss_weight = 1.0

        for t in range(len(hidden_states)):
            k_t = self.W_key(hidden_states[t])  # (dim,)
            v_t = self.W_val(hidden_states[t])  # (dim,)

            # Memory read: reconstruct v from k using M.
            v_pred_old = self.M @ k_t  # (dim,)

            # Self-supervised reconstruction loss.
            loss_t = F.mse_loss(v_pred_old, v_t)
            accumulated_loss += loss_t
            accumulated_steps += 1
            final_loss = loss_t.item()

            # Manual gradient of loss w.r.t. M.
            grad_M = torch.outer(v_pred_old - v_t, k_t)

            # Data-dependent gating per token (tensors, kept in graph).
            token_emb = hidden_states[t]
            alpha_tensor = self.gate_forget(token_emb).squeeze()  # scalar
            eta_tensor = self.gate_momentum(token_emb).squeeze()  # scalar
            theta_tensor = self.gate_lr(token_emb).squeeze()      # scalar

            # Use scalar values for the manual M/S update (no autograd).
            with torch.no_grad():
                alpha = alpha_tensor.item() * self.alpha_base
                eta = eta_tensor.item() * self.eta_base
                theta = theta_tensor.item() * self.theta_base

                # Titans momentum update.
                self.S = eta * self.S - theta * grad_M

                # Memory update with forgetting.
                self.M = (1 - alpha) * self.M + self.S

            # Surrogate gating loss: did this update improve retrieval?
            with torch.no_grad():
                v_pred_new = self.M @ k_t
                sim_old = F.cosine_similarity(
                    v_pred_old.unsqueeze(0), v_t.unsqueeze(0)
                )
                sim_new = F.cosine_similarity(
                    v_pred_new.unsqueeze(0), v_t.unsqueeze(0)
                )
                benefit = sim_new - sim_old  # scalar, [-2, 2]

                # If a surprise signal is provided, amplify the benefit term.
                if surprise is not None:
                    benefit = benefit + torch.tanh(
                        torch.tensor(surprise, device=self.device)
                    )

                # High benefit -> high update rate, high momentum, high LR.
                target = torch.sigmoid(benefit * 2.0)
                target = target.view_as(alpha_tensor)

            gate_loss_t = (
                F.binary_cross_entropy(alpha_tensor, target)
                + F.binary_cross_entropy(eta_tensor, target)
                + F.binary_cross_entropy(theta_tensor, target)
            )
            accumulated_loss += gate_loss_weight * gate_loss_t

            # Periodic end-to-end update of W_key/W_val and gating networks.
            if accumulated_steps % gate_period == 0:
                self.gate_optimizer.zero_grad()
                accumulated_loss.backward()
                self.gate_optimizer.step()
                accumulated_loss = 0.0

        # Leftover tokens that did not fill a full gate_period window.
        if accumulated_steps % gate_period != 0:
            self.gate_optimizer.zero_grad()
            accumulated_loss.backward()
            self.gate_optimizer.step()

        # Periodic capacity management.
        self._learn_count += len(hidden_states)
        if self._learn_count % 1000 < len(hidden_states):
            self.compress_memory()

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

    def compress_memory(self, threshold=0.01):
        """Zero out low-norm rows of M to manage capacity.

        Returns the number of rows compressed.
        """
        with torch.no_grad():
            row_norms = torch.norm(self.M, dim=1)  # (dim,)
            low_norm_mask = row_norms < threshold
            self.M[low_norm_mask] = 0.0
            compressed = low_norm_mask.sum().item()
        return compressed

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
                "gate_optimizer": self.gate_optimizer.state_dict(),
                "learn_count": self._learn_count,
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
        if "gate_optimizer" in state:
            self.gate_optimizer.load_state_dict(state["gate_optimizer"])
        if "learn_count" in state:
            self._learn_count = state["learn_count"]
