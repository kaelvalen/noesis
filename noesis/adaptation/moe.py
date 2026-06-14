"""Sparse Mixture-of-Experts output layer.

CRITICAL RULES:
- 32 slots are PRE-ALLOCATED. No runtime nn.ModuleList.append().
- Only a small number (default 4) are active in VRAM.
- Inactive experts are stored as state dicts on disk and lazy-loaded.
- Router: softmax + top-k masking.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoesisMoE(nn.Module):
    def __init__(self, d_model=1024, vocab_size=1024, num_experts=32, active=4, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.active = active
        self.device = device

        # Active experts in VRAM. Each expert is a small MLP that maps the
        # adapted hidden state directly to vocabulary logits.
        self.active_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, vocab_size),
                ).to(device)
                for _ in range(active)
            ]
        )

        # Inactive experts: state dicts on disk.
        self.inactive_dir = "noesis_experts/"
        os.makedirs(self.inactive_dir, exist_ok=True)
        for i in range(active, num_experts):
            path = f"{self.inactive_dir}/expert_{i}.pt"
            if not os.path.exists(path):
                dummy = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, vocab_size),
                )
                torch.save(dummy.state_dict(), path)

        self.router = nn.Linear(d_model, num_experts).to(device)
        self.register_buffer("expert_mask", torch.ones(num_experts))

    def forward(self, x):
        """x: (batch, seq, d_model). Returns logits of shape (batch, seq, vocab_size).

        Vectorized dispatch: tokens are flattened and routed to active experts
        in a single pass, avoiding per-token Python loops.
        """
        x = x.to(self.device)
        B, S, D = x.shape

        # Router: (B, S, num_experts)
        router_logits = self.router(x)
        masked = router_logits + (1 - self.expert_mask) * -1e9
        weights, indices = torch.topk(F.softmax(masked, dim=-1), self.active)
        # weights: (B, S, active), indices: (B, S, active)

        # Flatten for expert dispatch.
        x_flat = x.view(-1, D)  # (B*S, D)
        weights_flat = weights.view(-1, self.active)  # (B*S, active)
        indices_flat = indices.view(-1, self.active)  # (B*S, active)

        out_flat = x_flat.new_zeros(B * S, self.vocab_size)  # (B*S, V)

        # Dispatch each active expert over its assigned tokens.
        for expert_slot in range(len(self.active_experts)):
            # Tokens that routed to this expert slot in any of the top-k positions.
            mask = (indices_flat == expert_slot).any(dim=-1)  # (B*S,)
            if mask.sum() == 0:
                continue

            expert_input = x_flat[mask]  # (N, D)
            expert_output = self.active_experts[expert_slot](expert_input)  # (N, V)

            # Gather per-token weight for this expert slot.
            # weights_flat shape: (B*S, active); find the column matching this slot.
            matches = (indices_flat[mask] == expert_slot).float()  # (N, active)
            # Sum over active dimension (only one non-zero per row typically).
            expert_weights = (weights_flat[mask] * matches).sum(dim=-1)  # (N,)

            out_flat[mask] += expert_weights.unsqueeze(-1) * expert_output

        # Track how often each active expert is selected (for LRU eviction).
        if not hasattr(self, "expert_usage"):
            self.expert_usage = torch.zeros(
                len(self.active_experts), device=self.device
            )
        for expert_slot in range(len(self.active_experts)):
            mask = (indices_flat == expert_slot).any(dim=-1)
            self.expert_usage[expert_slot] += mask.sum().float()

        return out_flat.view(B, S, self.vocab_size)

    def swap_expert(self, slot_idx, state_dict_path):
        """Load an inactive expert state dict into an active slot."""
        if slot_idx < 0 or slot_idx >= self.active:
            warnings.warn(f"Invalid active slot index {slot_idx}")
            return False

        try:
            state = torch.load(state_dict_path, map_location=self.device)
            self.active_experts[slot_idx].load_state_dict(state)
            self.expert_mask[slot_idx] = 1.0
            return True
        except Exception as exc:
            warnings.warn(f"Failed to load expert from {state_dict_path}: {exc}")
            return False

    def load_expert_into_inactive_slot(self, expert_idx, state_dict_path):
        """Copy a state dict into the on-disk slot for an inactive expert."""
        if expert_idx < self.active or expert_idx >= self.num_experts:
            warnings.warn(f"Invalid inactive expert index {expert_idx}")
            return False
        try:
            state = torch.load(state_dict_path, map_location="cpu")
            torch.save(state, f"{self.inactive_dir}/expert_{expert_idx}.pt")
            return True
        except Exception as exc:
            warnings.warn(f"Failed to store expert {expert_idx}: {exc}")
            return False
