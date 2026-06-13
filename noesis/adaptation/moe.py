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


class NoesisMoE(nn.Module):
    def __init__(self, d_model=1024, num_experts=32, active=4, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.active = active
        self.device = device

        # Active experts in VRAM.
        self.active_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
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
                    nn.Linear(4 * d_model, d_model),
                )
                torch.save(dummy.state_dict(), path)

        self.router = nn.Linear(d_model, num_experts).to(device)
        self.register_buffer("expert_mask", torch.ones(num_experts))

    def forward(self, x):
        """x: (batch, seq, d_model). Returns output of same shape."""
        x = x.to(self.device)
        router_logits = self.router(x)
        masked = router_logits + (1 - self.expert_mask) * -1e9
        weights, indices = torch.topk(torch.softmax(masked, dim=-1), self.active)

        output = torch.zeros_like(x)
        batch_size = x.size(0)
        seq_len = x.size(1)

        for i in range(self.active):
            idx = indices[..., i]  # (batch, seq)
            w = weights[..., i].unsqueeze(-1)  # (batch, seq, 1)
            for b in range(batch_size):
                for s in range(seq_len):
                    expert_idx = idx[b, s].item()
                    if expert_idx < len(self.active_experts) and self.expert_mask[expert_idx]:
                        output[b, s] += w[b, s] * self.active_experts[expert_idx](x[b, s])
        return output

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
