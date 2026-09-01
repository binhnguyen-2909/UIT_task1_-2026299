# V25-SV: BGE-Gemma Semantic Swap Verifier

Code-only runner for the V25 legal-IR experiment. It evaluates conservative
slot-4/5 edits over the frozen V16b top five using raw legal evidence and
`BAAI/bge-reranker-v2-gemma`.

This repository deliberately contains no organizer data, model weights, public
labels, locked holdout, submission, E3B cache, raw block manifest, or partial
run output. See [docs/ASSETS_FROM_DRIVE.md](docs/ASSETS_FROM_DRIVE.md) for the
private files required to execute the experiment, and
[RUN_ON_SERVER.md](RUN_ON_SERVER.md) for the exact run order.

Before sharing any organizer-provided data with another person, verify that the
competition rules allow it. The server run must keep public labels and locked
holdout closed until the complete V25 OOF contract passes.

## Repository layout

- `scripts/`: V25 resolver, raw-evidence builder/scorer, and OOF verifier.
- `docs/V25_BGE_Gemma_Semantic_Swap_Verifier_PREREGISTRATION.md`: authoritative
  V25 preregistration.
- `docs/V25_SERVER_EXECUTION_SPEC.md`: concise server-run companion to the
  preregistration.
- `docs/RESOLVED_CONFIG.local-reference.json`: local audit reference only;
  create a fresh resolved config on the server because paths differ.

## Server requirements

- Python 3.10
- NVIDIA CUDA GPU with at least 24 GB VRAM recommended
- Enough disk for model, assets and generated outputs (at least 30 GB free)
- PyTorch matching the server CUDA runtime, plus
  `requirements-server.txt`

The exact model revision is:

```text
BAAI/bge-reranker-v2-gemma@1787044f8b6fb740a9de4557c3a12377f84d9e17
```
