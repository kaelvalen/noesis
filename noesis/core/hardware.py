"""Hardware abstraction layer for NOESIS.

Auto-detects available compute and memory resources and decides where each
component of the system should be placed. This is the single source of truth
for tensor/device placement in the codebase.
"""

import torch
import psutil


class HardwareManager:
    """Detects hardware and returns an optimal component placement policy."""

    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.vram_gb = 0.0
        if self.has_cuda:
            self.vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
        self.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.placement = self._optimize()

    def _optimize(self):
        """Return placement dict based on available VRAM/RAM."""
        if not self.has_cuda or self.vram_gb < 4:
            return {
                "backbone": "cpu",
                "ttt": "cpu",
                "moe": "cpu",
                "titans": "cpu",
                "cache": "ram",
                "vectordb": "disk",
            }
        elif self.vram_gb < 8:
            return {
                "backbone": "cuda",
                "ttt": "cuda",
                "moe": "cuda",
                "titans": "cpu",
                "cache": "ram",
                "vectordb": "disk",
            }
        elif self.vram_gb < 16:
            return {
                "backbone": "cuda",
                "ttt": "cuda",
                "moe": "cuda",
                "titans": "cuda",
                "cache": "cuda",
                "vectordb": "ram",
            }
        else:
            return {
                "backbone": "cuda",
                "ttt": "cuda",
                "moe": "cuda",
                "titans": "cuda",
                "cache": "cuda",
                "vectordb": "ram",
            }

    def move(self, tensor, target):
        """Move a tensor to the target device as decided by placement."""
        if target == "cuda" and self.has_cuda:
            return tensor.cuda()
        return tensor.cpu()

    def resolve_device(self, component):
        """Map a logical placement label to a valid torch device string.

        'ram' and 'disk' imply CPU-resident tensors; only 'cuda' maps to GPU.
        """
        target = self.placement.get(component, "cpu")
        if target == "cuda" and self.has_cuda:
            return "cuda"
        return "cpu"

    def summary(self):
        """Human-readable summary of detected hardware and placement."""
        lines = [
            f"CUDA available: {self.has_cuda}",
            f"VRAM: {self.vram_gb:.2f} GB",
            f"RAM: {self.ram_gb:.2f} GB",
            "Placement:",
        ]
        for component, location in self.placement.items():
            lines.append(f"  {component}: {location}")
        return "\n".join(lines)
