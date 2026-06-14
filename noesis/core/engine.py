"""End-to-end NOESIS inference and learning engine."""

import glob
import os
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F

from .backbone import NoesisBackbone, get_tokenizer
from .hardware import HardwareManager
from ..memory.titans import NoesisTitansLMM
from ..memory.sparse_cache import NoesisSparseCache
from ..memory.vectordb import NoesisVectorDB
from ..adaptation.ttt import NoesisTTT
from ..adaptation.moe import NoesisMoE
from ..adaptation.consolidator import NoesisConsolidator
from ..ingestion.web import ingest_web
from ..ingestion.local import ingest_local


class NoesisEngine:
    """Top-level NOESIS engine: frozen backbone + learning memory tiers."""

    def __init__(
        self,
        model_path: str = "BlinkDL/rwkv-5-world-1b5",
        use_mock: bool = False,
        d_model: int = 1024,
    ):
        self.hw = HardwareManager()
        self.device = self.hw.placement["backbone"]
        self.d_model = d_model

        # 1. Backbone (frozen, stateful).
        self.backbone = NoesisBackbone(model_path, self.device, use_mock=use_mock, d_model=d_model)
        self.tokenizer = get_tokenizer(model_path, use_mock=use_mock)

        # 2. TTT (sequence-local, train mode).
        self.ttt = NoesisTTT(d_model=d_model, device=self.hw.resolve_device("ttt"))

        # 3. Titans (persistent, CPU/RAM by default).
        self.titans = NoesisTitansLMM(dim=d_model, device=self.hw.resolve_device("titans"))
        if os.path.exists("noesis_state/titans.pt"):
            self.titans.load("noesis_state/titans.pt")

        # 4. Sparse Cache.
        self.cache = NoesisSparseCache(dim=d_model, device=self.hw.resolve_device("cache"))
        if os.path.exists("noesis_state/cache.pt"):
            self.cache.load("noesis_state/cache.pt")

        # 5. Vector DB.
        self.vectordb = NoesisVectorDB(dim=d_model)

        # 6. MoE (outputs logits directly).
        vocab_size = getattr(
            self.backbone.model, "vocab_size",
            getattr(self.tokenizer, "vocab_size", 1024)
        )
        self.moe = NoesisMoE(
            d_model=d_model, vocab_size=vocab_size, device=self.hw.resolve_device("moe")
        )

        # Session state.
        self.backbone_state = self._load_backbone_state()
        self.session_traces: List[dict] = []
        self.interaction_count = 0

        # Optional small embedding model for ingestion.
        self._embedder = None

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str):
        """Return a (1, seq_len) LongTensor on the backbone device."""
        result = self.tokenizer(text, return_tensors="pt")
        if isinstance(result, dict):
            return result["input_ids"].to(self.device)
        if isinstance(result, list):
            return torch.tensor([result], dtype=torch.long, device=self.device)
        return result.to(self.device)

    def _decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        # RWKV PIPELINE decode accepts list.
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_backbone_state(self):
        path = "noesis_state/backbone_state.pt"
        if os.path.exists(path):
            try:
                return torch.load(path, map_location=self.device)
            except Exception as exc:
                warnings.warn(f"Failed to load backbone state: {exc}")
        return None

    def _autosave(self):
        os.makedirs("noesis_state", exist_ok=True)
        if self.backbone_state is not None:
            torch.save(self.backbone_state, "noesis_state/backbone_state.pt")
        self.titans.save("noesis_state/titans.pt")
        self.cache.save("noesis_state/cache.pt")
        torch.save(self.session_traces, "noesis_state/traces.pt")

    # ------------------------------------------------------------------
    # Main interaction loop
    # ------------------------------------------------------------------

    def interact(
        self,
        user_input: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """Process a user turn, learn, and generate a response."""
        # 1. Tokenize.
        input_ids = self._encode(user_input)

        # 2. Retrieve from all memory tiers.
        query_emb = self.backbone.get_hidden_states(input_ids, self.backbone_state)[
            0, -1
        ]

        # L2: Titans associative.
        titans_val, titans_conf = self.titans.retrieve(query_emb.cpu())

        # L3: Sparse exact.
        sparse_vals, sparse_scores = self.cache.retrieve(query_emb.cpu(), k=3)

        # L4: Vector DB semantic.
        vectordb_hits = self.vectordb.search(query_emb.cpu(), k=5)

        # 3. Augment input with memory context.
        memory_text = self._format_memory(titans_val, sparse_vals, vectordb_hits)
        if memory_text:
            augmented_input = f"[Memory: {memory_text}]\n{user_input}"
        else:
            augmented_input = user_input

        aug_ids = self._encode(augmented_input)

        # Reset sequence-local TTT fast weights at the start of each interaction.
        self.ttt.reset()

        # 4. Backbone forward (stateful, frozen) — return logits + hidden states.
        logits, self.backbone_state, hidden = self.backbone.forward_with_hidden(
            aug_ids, self.backbone_state
        )

        # 5. TTT adaptation (train mode, inner loop) on d_model hidden states.
        labels = hidden.roll(shifts=-1, dims=1)
        ttt_out, surprise = self.ttt(hidden, labels)

        # 6. Titans learn (CPU, persistent).
        # TTT output is already detached (gradient barrier); no extra detach needed.
        ttt_cpu = ttt_out[0].cpu()  # (seq, dim) on CPU
        titans_loss = self.titans.learn(ttt_cpu, surprise=surprise)

        # 7. Sparse cache write (if surprising).
        last_k = self.titans.W_key(ttt_cpu[-1])
        last_v = self.titans.W_val(ttt_cpu[-1])
        self.cache.write(last_k, last_v, surprise)

        # 8. MoE output (GPU/CPU per HAL).
        logits = self.moe(ttt_out)

        # 9. Generate with sampling.
        generated = self._generate(
            logits,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        output_text = self._decode(generated)

        # 10. Trace and auto-save.
        # Store post-TTT hidden states and the actual token ids so the
        # consolidator can learn a real logit mapping instead of an identity.
        self.session_traces.append(
            {
                "input": user_input,
                "output": output_text,
                "surprise": surprise,
                "titans_loss": titans_loss,
                "backbone_hidden": hidden[0, -1].detach().cpu().clone(),
                "ttt_out": ttt_cpu.clone(),
                "input_ids": aug_ids[0].detach().cpu().clone(),
            }
        )
        self.interaction_count += 1
        if self.interaction_count % 10 == 0:
            self._autosave()

        return output_text

    def _generate(
        self,
        logits,
        max_new_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    ):
        """Sample token ids from logits with temperature, top-k, and top-p."""
        generated = []
        eos_id = getattr(self.tokenizer, "eos_token_id", 1)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]  # (batch, vocab)

            if temperature > 0.0:
                next_logits = next_logits / temperature

                # Top-k filtering.
                if top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    kth_vals = torch.topk(next_logits, top_k)[0][..., -1, None]
                    indices_to_remove = next_logits < kth_vals
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, -float("inf")
                    )

                # Top-p (nucleus) filtering.
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, -float("inf")
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated.append(next_token.item())
            if next_token.item() == eos_id:
                break

            # Re-run backbone for next token.
            logits, self.backbone_state, hidden = self.backbone.forward_with_hidden(
                next_token, self.backbone_state
            )
            # Use frozen TTT during generation: do not mutate sequence-local W_fast.
            ttt_out = self.ttt.forward_eval(hidden)
            logits = self.moe(ttt_out)

        return generated

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_web(self, urls: List[str]):
        """Fetch URLs and store chunks in the Vector DB."""
        for url in urls:
            text = ingest_web(url)
            if text:
                chunks = self._chunk_text(text)
                embs = self._embed(chunks)
                self.vectordb.ingest(chunks, embs)

    def ingest_files(self, paths: List[str]):
        """Extract text from files/globs and store chunks in the Vector DB."""
        all_files = []
        for pattern in paths:
            all_files.extend(glob.glob(pattern))
        for path in all_files:
            text = ingest_local(path)
            if text:
                chunks = self._chunk_text(text)
                embs = self._embed(chunks)
                self.vectordb.ingest(chunks, embs)

    def _embed(self, chunks: List[str]):
        """Return embeddings for chunks.

        If a sentence-transformer model is available, use it. Otherwise fall
        back to the backbone hidden-state mean for each chunk.
        """
        if not chunks:
            return torch.empty(0, self.d_model)

        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedder = "backbone"

        if self._embedder != "backbone":
            embs = self._embedder.encode(chunks, convert_to_tensor=True)
            return embs.detach().cpu()

        # Fallback: use backbone hidden states.
        embeddings = []
        for chunk in chunks:
            ids = self._encode(chunk)
            with torch.no_grad():
                h = self.backbone.get_hidden_states(ids, self.backbone_state)
            embeddings.append(h[0].mean(dim=0).cpu())
        return torch.stack(embeddings)

    def _chunk_text(self, text: str, size: int = 512):
        return [text[i : i + size] for i in range(0, len(text), size)]

    # ------------------------------------------------------------------
    # Memory formatting and consolidation
    # ------------------------------------------------------------------

    def _format_memory(self, titans_val, sparse_vals, vectordb_hits):
        parts = []
        if vectordb_hits:
            parts.append(
                "Docs: " + "; ".join([h[0][:200] for h in vectordb_hits[:3]])
            )
        return " | ".join(parts) if parts else ""

    def consolidate(self):
        consolidator = NoesisConsolidator(self)
        return consolidator.run()

    def save(self):
        """Manual save of all persistent state."""
        self._autosave()

    def status(self):
        """Return a status dictionary for CLI/status commands."""
        import psutil

        status = {
            "hardware": self.hw.summary(),
            "backbone": self.backbone.backend_name(),
            "titans_M_shape": list(self.titans.M.shape),
            "cache_fill": self.cache.write_ptr,
            "cache_capacity": self.cache.capacity,
            "moe_active": self.moe.active,
            "moe_total": self.moe.num_experts,
            "interactions": self.interaction_count,
            "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        }
        if self.hw.has_cuda:
            status["vram_used_gb"] = (
                torch.cuda.memory_allocated() / (1024 ** 3)
            )
        return status
