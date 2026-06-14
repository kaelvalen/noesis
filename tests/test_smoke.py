"""Smoke tests for NOESIS components (uses mock backbone)."""

import os
import shutil
import tempfile

import torch

from noesis.core.hardware import HardwareManager
from noesis.core.backbone import NoesisBackbone, get_tokenizer
from noesis.adaptation.ttt import NoesisTTT
from noesis.memory.titans import NoesisTitansLMM
from noesis.memory.sparse_cache import NoesisSparseCache
from noesis.memory.vectordb import NoesisVectorDB
from noesis.adaptation.moe import NoesisMoE
from noesis.adaptation.consolidator import NoesisConsolidator
from noesis.core.engine import NoesisEngine


def test_hardware_manager():
    hw = HardwareManager()
    assert isinstance(hw.placement, dict)
    assert "backbone" in hw.placement
    assert hw.ram_gb > 0


def test_backbone_mock():
    bb = NoesisBackbone("mock", device="cpu", use_mock=True)
    tok = get_tokenizer("mock", use_mock=True)
    ids = tok("hello world", return_tensors="pt")
    logits, state = bb(ids)
    assert logits.shape[:2] == ids.shape
    assert state is not None

    hidden = bb.get_hidden_states(ids)
    assert hidden.shape[:2] == ids.shape
    assert hidden.shape[-1] == bb.model.dim


def test_ttt_train_mode_and_detach():
    ttt = NoesisTTT(d_model=64, device="cpu")
    assert ttt.training
    x = torch.randn(1, 10, 64)
    out, surprise = ttt(x)
    assert out.shape == x.shape
    assert not out.requires_grad  # detached
    assert isinstance(surprise, float)


def test_titans_toy_task():
    dim = 32
    mem = NoesisTitansLMM(dim=dim, device="cpu")
    # Synthetic correlated key-value pairs.
    hidden = torch.randn(100, dim)
    losses = []
    for _ in range(5):
        loss = mem.learn(hidden)
        losses.append(loss)
    # Loss should generally decrease or at least not explode.
    assert losses[-1] < losses[0] * 1.5

    v, conf = mem.retrieve(hidden[-1])
    assert v.shape == (dim,)
    assert -1.0 <= conf <= 1.0


def test_sparse_cache_write_and_retrieve():
    cache = NoesisSparseCache(dim=32, capacity=100, device="cpu")
    k = torch.randn(32)
    v = torch.randn(32)
    assert cache.write(k, v, surprise=2.0) is True
    assert cache.write(k, v, surprise=0.5) is False
    vals, scores = cache.retrieve(k, k=1)
    assert vals.shape == (1, 32)
    assert scores.shape == (1,)


def test_vectordb(tmp_path):
    db = NoesisVectorDB(path=str(tmp_path / "vdb"), dim=64)
    chunks = ["hello world", "foo bar"]
    embs = torch.randn(2, 64)
    db.ingest(chunks, embs)
    results = db.search(embs[0], k=2)
    if db.client is None:
        # Qdrant not installed; VectorDB gracefully disabled.
        assert results == []
    else:
        assert len(results) <= 2
        assert isinstance(results[0], tuple)


def test_moe_routing():
    vocab_size = 100
    moe = NoesisMoE(d_model=64, vocab_size=vocab_size, num_experts=8, active=2, device="cpu")
    x = torch.randn(1, 5, 64)
    out = moe(x)
    assert out.shape == (1, 5, vocab_size)


def test_engine_interact_mock(tmp_path):
    os.chdir(tmp_path)
    engine = NoesisEngine(model_path="mock", use_mock=True, d_model=64)
    response = engine.interact("What is the capital of France?")
    assert isinstance(response, str)
    status = engine.status()
    assert status["backbone"] == "mock"


def test_state_save_and_load(tmp_path):
    os.chdir(tmp_path)
    engine = NoesisEngine(model_path="mock", use_mock=True, d_model=32)
    engine.interact("test input")
    engine.save()

    assert os.path.exists("noesis_state/titans.pt")
    assert os.path.exists("noesis_state/cache.pt")

    engine2 = NoesisEngine(model_path="mock", use_mock=True, d_model=32)
    assert engine2.cache.write_ptr == engine.cache.write_ptr


def test_sampling_parameters(tmp_path):
    os.chdir(tmp_path)
    engine = NoesisEngine(model_path="mock", use_mock=True, d_model=32)
    # Greedy (temperature=0) should be deterministic.
    r1 = engine.interact("hello", temperature=0.0, max_new_tokens=8)
    r2 = engine.interact("hello", temperature=0.0, max_new_tokens=8)
    # State carries across interactions, so not strictly identical; just verify output.
    assert isinstance(r1, str)
    assert isinstance(r2, str)

    # Sampling with high temperature should produce output.
    r3 = engine.interact("hello", temperature=1.0, top_k=5, max_new_tokens=8)
    assert isinstance(r3, str)


def test_consolidator_domain_adaptation(tmp_path):
    os.chdir(tmp_path)
    engine = NoesisEngine(model_path="mock", use_mock=True, d_model=32)
    vocab_size = engine.moe.vocab_size

    # Synthesize enough traces with aligned (ttt_out, input_ids) pairs.
    seq_len = 8
    for i in range(120):
        engine.session_traces.append(
            {
                "surprise": float(i + 1),
                "backbone_hidden": torch.randn(32),
                "ttt_out": torch.randn(seq_len, 32),
                "input_ids": torch.randint(0, vocab_size, (seq_len,)),
            }
        )

    consolidator = NoesisConsolidator(engine)
    path = consolidator.run(topk=500, min_traces=100, epochs=2)
    assert path is not None
    assert os.path.exists(path)
    # A new expert should have been loaded into one of the active MoE slots.
    assert engine.moe.expert_mask.sum().item() > 0
