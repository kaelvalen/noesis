# NOESIS

Memory-centric, self-improving inference system.

## Philosophy

- **Frozen backbone**: RWKV-5 / Mamba small language model (1B–1.5B), never fine-tuned.
- **External memory**: Titans LMM, Sparse Residual Cache, and Qdrant Vector DB store all knowledge.
- **Inference-time learning**: Test-Time Training (TTT) adapts per sequence; Titans updates associative memory during the forward pass.
- **Hardware abstraction**: Automatically places components on CPU/GPU/RAM/disk based on available VRAM.

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

For real model support, also install the backbone of your choice:

```bash
pip install rwkv        # RWKV-5 preferred
# or
pip install mamba-ssm   # Mamba alternative
```

## Quick start (mock backbone — no large download)

```bash
noesis --mock init
noesis --mock chat
noesis --mock status
```

## Quick start (RWKV-5-1.5B)

```bash
noesis init --model BlinkDL/rwkv-5-world-1b5
noesis chat
```

## CLI commands

- `noesis init --model <path>` — initialize persistent state.
- `noesis chat` — stateful chat that learns during conversation.
  - `--max-tokens N`
  - `--temperature T` (0.0 = greedy)
  - `--top-k K`
  - `--top-p P` (nucleus sampling)
- `noesis learn-web <url> [<url> ...]` — ingest web pages into Vector DB.
- `noesis learn-files <glob> [<glob> ...]` — ingest local files.
- `noesis status` — show hardware placement, memory fill, and usage.
- `noesis save` — manually checkpoint all state.
- `noesis consolidate` — train a new MoE expert from high-surprise traces.

## Project structure

```
noesis/
├── noesis/
│   ├── core/         # Engine, hardware, backbone
│   ├── memory/       # Titans, sparse cache, vector DB
│   ├── adaptation/   # TTT, MoE, consolidation
│   ├── ingestion/    # Web / file ingestion
│   └── cli.py        # CLI entry point
├── tests/            # Smoke tests
├── requirements.txt
└── setup.py
```

## Testing

```bash
pytest tests/test_smoke.py -v
```

## Notes

- The backbone weights are **never updated**. All learning occurs in Titans `M`, the sparse cache, MoE adapters, and the vector DB.
- State is auto-saved every 10 interactions to `noesis_state/`.
- The mock backbone is intended for development and smoke testing only.
