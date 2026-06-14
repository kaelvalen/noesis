"""Backbone model wrapper for NOESIS.

Prefers RWKV-5 (O(1) recurrent state) and falls back to Mamba or a tiny
mock backbone for development/testing when the heavy dependencies are not
installed or the weights are not present.

Golden Rule: the backbone is ALWAYS frozen and run in eval() mode.
"""

import os
import re
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _resolve_dtype(device: str):
    """Prefer bfloat16, fall back to float16 or float32 depending on device."""
    if device == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Tiny deterministic tokenizer for development without model weights."""

    PAD_ID = 0
    EOS_ID = 1
    BOS_ID = 2
    UNK_ID = 3

    def __init__(self):
        self.vocab = {"<pad>": 0, "</s>": 1, "<s>": 2, "<unk>": 3}
        self.reverse = {v: k for k, v in self.vocab.items()}
        self._next_id = 4

    def _token(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = self._next_id
            self.reverse[self._next_id] = word
            self._next_id += 1
        return self.vocab[word]

    def encode(self, text: str, return_tensors: Optional[str] = None):
        tokens = [
            self.BOS_ID,
            *[
                self._token(tok)
                for tok in re.findall(r"\w+|[^\w\s]", text.strip())
                if tok.strip()
            ],
            self.EOS_ID,
        ]
        tensor = torch.tensor([tokens], dtype=torch.long)
        if return_tensors == "pt":
            return tensor
        return tokens

    def __call__(self, text: str, return_tensors: Optional[str] = None):
        return self.encode(text, return_tensors=return_tensors)

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        parts = []
        for i in ids:
            tok = self.reverse.get(i, "<unk>")
            if skip_special_tokens and tok in ("<pad>", "</s>", "<s>", "<unk>"):
                continue
            parts.append(tok)
        return " ".join(parts)

    @property
    def eos_token_id(self):
        return self.EOS_ID

    @property
    def pad_token_id(self):
        return self.PAD_ID

    @property
    def vocab_size(self):
        return len(self.vocab)


def get_tokenizer(model_path: Optional[str] = None, use_mock: bool = False):
    """Load a tokenizer matching the backbone."""
    if use_mock:
        return _MockTokenizer()

    # RWKV ships its own tokenizer utilities.
    try:
        from rwkv.utils import PIPELINE

        return PIPELINE(model_path, "rwkv_vocab_v20230424")
    except Exception:
        pass

    # Generic HuggingFace tokenizer if the model is on HF.
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        pass

    warnings.warn(
        "No specialized tokenizer found; falling back to mock tokenizer. "
        "Install rwkv or transformers for real model support."
    )
    return _MockTokenizer()


# ---------------------------------------------------------------------------
# Backbone implementations
# ---------------------------------------------------------------------------


class _MockBackbone(nn.Module):
    """Small trainable-but-frozen LM head used when no real model is available."""

    def __init__(self, vocab_size: int, dim: int = 1024, device: str = "cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.device = device
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRUCell(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.to(device)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, tokens, state=None):
        # tokens: (batch, seq_len)
        batch, seq = tokens.shape
        if state is None:
            state = torch.zeros(batch, self.dim, device=self.device)
        hidden_list = []
        for t in range(seq):
            emb = self.embedding(tokens[:, t])
            state = self.rnn(emb, state)
            hidden_list.append(state)
        hidden = torch.stack(hidden_list, dim=1)  # (batch, seq, dim)
        logits = self.lm_head(hidden)
        return logits, state

    def forward_with_hidden(self, tokens, state=None):
        """Return logits, new state, and pre-output hidden states."""
        batch, seq = tokens.shape
        if state is None:
            state = torch.zeros(batch, self.dim, device=self.device)
        hidden_list = []
        for t in range(seq):
            emb = self.embedding(tokens[:, t])
            state = self.rnn(emb, state)
            hidden_list.append(state)
        hidden = torch.stack(hidden_list, dim=1)  # (batch, seq, dim)
        logits = self.lm_head(hidden)
        return logits, state, hidden

    def get_hidden_states(self, tokens, state=None):
        """Return hidden states without mutating recurrent state."""
        with torch.no_grad():
            batch, seq = tokens.shape
            if state is None:
                state = torch.zeros(batch, self.dim, device=self.device)
            hidden_list = []
            for t in range(seq):
                emb = self.embedding(tokens[:, t])
                state = self.rnn(emb, state)
                hidden_list.append(state.clone())
            hidden = torch.stack(hidden_list, dim=1)
        return hidden


class NoesisBackbone(torch.nn.Module):
    """Frozen, stateful backbone wrapper.

    Args:
        model_path: Path or HF-id of the backbone weights.
        device: "cpu" or "cuda".
        use_mock: If True, load the tiny mock backbone for testing.
    """

    def __init__(self, model_path: str, device: str, use_mock: bool = False, d_model: int = 1024):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.dtype = _resolve_dtype(device)
        self.d_model = d_model
        self._backend_name = "mock"
        self.model = None

        if use_mock:
            warnings.warn("Using MOCK backbone for development/testing.")
            self.model = _MockBackbone(vocab_size=1024, dim=d_model, device=device)
            self._backend_name = "mock"
        else:
            self._try_load_rwkv(model_path, device)
            if self.model is None:
                self._try_load_mamba(model_path, device)
            if self.model is None:
                warnings.warn(
                    "Failed to load RWKV/Mamba; falling back to mock backbone."
                )
                self.model = _MockBackbone(vocab_size=1024, dim=d_model, device=device)
                self._backend_name = "mock"

        # Golden Rule: backbone is frozen and in eval mode.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.to(device)

    def _try_load_rwkv(self, model_path: str, device: str):
        try:
            from rwkv.model import RWKV

            strategy = device
            if device == "cuda":
                strategy = f"{device} bf16" if self.dtype == torch.bfloat16 else f"{device} fp16"
            self.model = RWKV(model=model_path, strategy=strategy)
            self._backend_name = "rwkv"
            return True
        except Exception as exc:
            warnings.warn(f"RWKV load failed: {exc}")
            return False

    def _try_load_mamba(self, model_path: str, device: str):
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

            self.model = MambaLMHeadModel.from_pretrained(
                model_path, device=device, dtype=self.dtype
            )
            self._backend_name = "mamba"
            return True
        except Exception as exc:
            warnings.warn(f"Mamba load failed: {exc}")
            return False

    def forward(self, tokens, state=None):
        """Return logits and new state."""
        tokens = tokens.to(self.device)
        with torch.no_grad():
            if self._backend_name == "rwkv":
                logits, new_state = self.model.forward(tokens, state)
            elif self._backend_name == "mamba":
                out = self.model(tokens, inference_params=state)
                logits = out.logits
                new_state = out  # carry full output for stateful mamba
            else:
                logits, new_state = self.model(tokens, state)
        return logits, new_state

    def forward_with_hidden(self, tokens, state=None):
        """Return logits, new state, and (batch, seq, d_model) hidden states."""
        tokens = tokens.to(self.device)
        with torch.no_grad():
            if self._backend_name == "mock":
                logits, new_state, hidden = self.model.forward_with_hidden(tokens, state)
            elif self._backend_name in ("rwkv", "mamba"):
                logits, new_state, hidden = self._forward_with_hook(tokens, state)
            else:
                logits, new_state = self.forward(tokens, state)
                hidden = self._approximate_hidden_from_logits(logits)
        return logits, new_state, hidden

    def get_hidden_states(self, tokens, state=None):
        """Return (batch, seq, d_model) hidden states before the output head."""
        with torch.no_grad():
            if self._backend_name == "mock":
                return self.model.get_hidden_states(tokens.to(self.device), state)

            if self._backend_name in ("rwkv", "mamba"):
                _, new_state, hidden = self._forward_with_hook(tokens, state)
                return hidden

            logits, _ = self.forward(tokens, state)
            hidden = self._approximate_hidden_from_logits(logits)
        return hidden

    def _find_head_module(self):
        """Locate the final output projection module to hook for hidden states."""
        for name, module in self.model.named_modules():
            if name.split(".")[-1] in ("head", "lm_head", "output"):
                return module
        return None

    def _forward_with_hook(self, tokens, state=None):
        """Run a forward pass and capture the input to the output head.

        For both RWKV and Mamba the pre-head hidden state is the input to the
        final linear projection. A forward hook is the most backend-agnostic
        way to retrieve it without forking model code.
        """
        head = self._find_head_module()
        if head is None:
            warnings.warn(
                f"Could not locate output head module for backend '{self._backend_name}'; "
                "falling back to logit-heuristic hidden approximation. "
                "Production deployments should verify hidden-state extraction."
            )
            logits, new_state = self.forward(tokens, state)
            return logits, new_state, self._approximate_hidden_from_logits(logits)

        captured = {}

        def _hook(module, inputs, output):
            if isinstance(inputs, tuple) and inputs[0] is not None:
                captured["hidden"] = inputs[0].detach().clone()

        handle = head.register_forward_hook(_hook)
        try:
            if self._backend_name == "mamba":
                out = self.model(tokens, inference_params=state)
                logits = out.logits
                new_state = out
            else:  # rwkv
                logits, new_state = self.model.forward(tokens, state)
        finally:
            handle.remove()

        if "hidden" not in captured:
            warnings.warn(
                "Output-head hook did not capture a hidden state; "
                "falling back to logit-heuristic hidden approximation."
            )
            return logits, new_state, self._approximate_hidden_from_logits(logits)

        return logits, new_state, captured["hidden"]

    def _approximate_hidden_from_logits(self, logits):
        """Approximate pre-output hidden states from logits (best-effort).

        For a linear output head, logits = hidden @ W_out.T.  We estimate
        hidden ≈ normalized_logits @ W_out, where normalized_logits are
        logits with the per-token max subtracted for numerical stability.

        This is a heuristic. Production RWKV deployments should patch the
        backbone's forward() to return hidden states directly; see
        patches/rwkv_hidden.patch.
        """
        W = self._output_weight()
        if W is None or W.shape[0] != logits.shape[-1] or W.shape[1] != self.d_model:
            # Last-resort: logits themselves as pseudo-hidden.
            return logits

        # Stable logits: subtract max per position.
        norm_logits = logits - logits.amax(dim=-1, keepdim=True)
        hidden = torch.einsum("b s v, v d -> b s d", norm_logits, W)
        return hidden

    def _output_weight(self):
        """Locate the output projection weight for hidden-state approximation."""
        candidates = [
            ("get_header_weight", lambda m: m.get_header_weight()),
            ("head.weight", lambda m: getattr(m, "head", None).weight.data if hasattr(m, "head") else None),
            ("lm_head.weight", lambda m: getattr(m, "lm_head", None).weight.data if hasattr(m, "lm_head") else None),
        ]
        # RWKV stores parameters in self.model.w as a dict.
        if hasattr(self.model, "w") and isinstance(self.model.w, dict):
            for key in ("head.weight", "output.weight", "lm_head.weight"):
                if key in self.model.w:
                    return self.model.w[key]

        for name, getter in candidates:
            try:
                W = getter(self.model)
                if W is not None:
                    return W
            except Exception:
                continue
        return None

    def backend_name(self) -> str:
        return self._backend_name
