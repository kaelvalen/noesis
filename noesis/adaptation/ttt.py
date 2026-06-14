"""Test-Time Training (TTT) linear layer.

CRITICAL IMPLEMENTATION RULES:
- MUST be in train() mode so the inner-loop gradient updates are active.
- Fast weights W_fast are sequence-local, NOT persistent.
- Self-supervised loss predicts the next hidden state from the current one.
- Output MUST be detach()-ed before passing to Titans LMM (gradient barrier).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoesisTTT(nn.Module):
    def __init__(self, d_model=1024, device="cuda", lr=0.01):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.lr = lr

        # Fixed projection (small, always loaded).
        self.W_proj = nn.Linear(d_model, d_model).to(device)

        # Sequence-local fast weights; recreated per session.
        self.W_fast = None
        self.train()  # TTT MUST run in train mode.

    def init_sequence(self, batch_size=1, dtype=torch.float32):
        """Initialize (or reset) fast weights for a new sequence."""
        self.W_fast = torch.zeros(
            batch_size,
            self.d_model,
            self.d_model,
            device=self.device,
            dtype=dtype,
        )

    def reset(self):
        """Alias for init_sequence(batch_size=1)."""
        self.init_sequence(batch_size=1)

    def forward(self, hidden_states, labels=None):
        """
        hidden_states: (batch, seq, d_model) from backbone. GPU.
        labels: hidden_states rolled by -1 (self-supervised next-step prediction).
        Returns: (adapted_hidden, surprise_metric)
        """
        hidden_states = hidden_states.to(self.device)
        if (
            self.W_fast is None
            or self.W_fast.size(0) != hidden_states.size(0)
            or self.W_fast.dtype != hidden_states.dtype
        ):
            self.init_sequence(hidden_states.size(0), dtype=hidden_states.dtype)

        if labels is None:
            labels = hidden_states.roll(shifts=-1, dims=1)

        losses = []
        seq_len = hidden_states.size(1)

        # TTT inner loop: 1 manual SGD step per token.
        for t in range(seq_len):
            x_t = hidden_states[:, t]  # (batch, d_model)
            y_t = labels[:, t]  # (batch, d_model)

            # Prediction using fast weights.
            pred = torch.bmm(x_t.unsqueeze(1), self.W_fast).squeeze(1)
            loss = F.mse_loss(pred, y_t, reduction="none").mean(dim=1)
            losses.append(loss)

            # Manual gradient of loss w.r.t W_fast.
            grad = torch.bmm(
                (pred - y_t).unsqueeze(2),  # (batch, d_model, 1)
                x_t.unsqueeze(1),  # (batch, 1, d_model)
            )  # (batch, d_model, d_model)

            # SGD update on fast weights.
            self.W_fast = self.W_fast - self.lr * grad

        # Apply adapted fast weights to full sequence.
        adapted = torch.bmm(hidden_states, self.W_fast)

        # Surprise metric: average reconstruction loss.
        with torch.no_grad():
            surprise = torch.stack(losses).mean().item()

        # GRADIENT BARRIER: detach before Titans.
        return adapted.detach(), surprise

    def forward_eval(self, hidden_states):
        """Inference-time TTT: apply current W_fast without updating it.

        Use this during autoregressive generation so that the sequence-local
        fast weights are not further mutated by generated tokens.
        """
        hidden_states = hidden_states.to(self.device)
        if (
            self.W_fast is None
            or self.W_fast.size(0) != hidden_states.size(0)
            or self.W_fast.dtype != hidden_states.dtype
        ):
            self.init_sequence(hidden_states.size(0), dtype=hidden_states.dtype)

        # Apply adapted fast weights without inner-loop updates.
        adapted = torch.bmm(hidden_states, self.W_fast)
        return adapted.detach()
