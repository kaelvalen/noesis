"""Nightly/background consolidation pipeline.

When enough interactions are accumulated, train a small adapter on the
highest-surprise traces and add it to the MoE as a new expert.
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

    def run(self, topk=500, min_traces=100, epochs=3):
        """Train an adapter on high-surprise traces and install it as an expert."""
        traces = sorted(
            self.engine.session_traces, key=lambda x: x.get("surprise", 0.0), reverse=True
        )[:topk]

        if len(traces) < min_traces:
            warnings.warn(
                f"Consolidation requires at least {min_traces} traces; got {len(traces)}."
            )
            return None

        d_model = self.engine.ttt.d_model
        adapter = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, d_model)
        )
        opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for tr in traces:
                ttt_out = tr.get("ttt_out")
                if ttt_out is None:
                    continue
                target = ttt_out.detach()
                pred = adapter(target)
                loss = F.mse_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            print(f"Consolidation epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

        # Save to disk.
        os.makedirs("noesis_experts", exist_ok=True)
        path = f"noesis_experts/new_expert_{int(time.time())}.pt"
        torch.save(adapter.state_dict(), path)

        # Activate in MoE.
        moe = self.engine.moe
        empty = (moe.expert_mask == 0).nonzero(as_tuple=False)
        if len(empty) > 0:
            moe.swap_expert(empty[0].item(), path)
        else:
            # Replace lowest-usage expert (LRU heuristic: first active slot).
            moe.swap_expert(0, path)

        # Clear traces.
        self.engine.session_traces = []
        return path
