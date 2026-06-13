"""Sparse Residual Cache.

Stores incompressible, high-surprise tokens exactly. RAM-backed with LRU
eviction to disk when capacity is exceeded.
"""

import os
import time
import torch
import torch.nn.functional as F


class NoesisSparseCache:
    def __init__(self, dim=1024, capacity=100000, device="cpu"):
        self.dim = dim
        self.capacity = capacity
        self.device = device
        self.threshold = 1.5

        # RAM store.
        self.keys = torch.zeros(capacity, dim, device=device)
        self.vals = torch.zeros(capacity, dim, device=device)
        self.surprise_scores = torch.zeros(capacity, device=device)
        self.lru_counter = torch.zeros(capacity, device=device)
        self.write_ptr = 0

        # Disk spill directory.
        self.spill_dir = "noesis_cache_spill/"
        os.makedirs(self.spill_dir, exist_ok=True)

    def write(self, key, value, surprise):
        """Write (key, value) if surprise exceeds threshold."""
        if surprise < self.threshold:
            return False

        key = key.detach().to(self.device)
        value = value.detach().to(self.device)

        if self.write_ptr >= self.capacity:
            # LRU eviction to disk.
            lru_idx = torch.argmin(self.lru_counter).item()
            self._spill_to_disk(lru_idx)
            self.write_ptr = lru_idx

        self.keys[self.write_ptr] = key
        self.vals[self.write_ptr] = value
        self.surprise_scores[self.write_ptr] = surprise
        self.lru_counter[self.write_ptr] = torch.max(self.lru_counter) + 1
        self.write_ptr += 1
        return True

    def retrieve(self, query, k=5):
        """Return top-k values and scores for a query key."""
        if self.write_ptr == 0:
            return torch.empty(0, self.dim), torch.empty(0)

        query = query.to(self.device)
        scores = F.cosine_similarity(
            query.unsqueeze(0), self.keys[: self.write_ptr], dim=1
        )
        topk = torch.topk(scores, min(k, self.write_ptr))

        # Update LRU counters for retrieved items.
        self.lru_counter[topk.indices] = torch.max(self.lru_counter) + 1

        return self.vals[topk.indices], topk.values

    def _spill_to_disk(self, idx):
        item = {
            "key": self.keys[idx].cpu(),
            "val": self.vals[idx].cpu(),
            "surprise": self.surprise_scores[idx].item(),
        }
        path = f"{self.spill_dir}/spill_{idx}_{int(time.time())}.pt"
        torch.save(item, path)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "keys": self.keys,
                "vals": self.vals,
                "scores": self.surprise_scores,
                "lru": self.lru_counter,
                "ptr": self.write_ptr,
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.keys = state["keys"].to(self.device)
        self.vals = state["vals"].to(self.device)
        self.surprise_scores = state["scores"].to(self.device)
        self.lru_counter = state["lru"].to(self.device)
        self.write_ptr = state["ptr"]
