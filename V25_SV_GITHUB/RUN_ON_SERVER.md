# Run V25-SV on a server

Run every command from the repository root. Do not copy local partial pack
files or mix them with server output.

## 1. Install the environment

Install the CUDA-compatible PyTorch build for the server first, then run:

```bash
python -m pip install -r requirements-server.txt
```

## 2. Place private assets

Copy the permitted Drive assets into these paths:

```text
assets/v18b_metadata.json
assets/v23_b0_report.json
assets/v23_b0_edit_rows.jsonl
assets/fold_manifest.json
assets/legalir.sqlite3
```

Do not copy public labels, locked holdout, submission files, or any local
`*.tmp` partial pack file.

## 3. Download the exact model revision

If the server can access Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-reranker-v2-gemma', revision='1787044f8b6fb740a9de4557c3a12377f84d9e17', local_dir='models/bge-reranker-v2-gemma-1787044f')"
```

If it cannot, copy the complete model directory privately to
`models/bge-reranker-v2-gemma-1787044f`.

## 4. Resolve and validate the recipe

```bash
python scripts/v25_sv_resolve_config.py \
  --prereg docs/V25_BGE_Gemma_Semantic_Swap_Verifier_PREREGISTRATION.md \
  --model-dir models/bge-reranker-v2-gemma-1787044f \
  --metadata assets/v18b_metadata.json \
  --v23-b0-report assets/v23_b0_report.json \
  --v23-b0-edits assets/v23_b0_edit_rows.jsonl \
  --fold-manifest assets/fold_manifest.json \
  --contexts-db assets/legalir.sqlite3 \
  --output-dir output/v25_sv
```

## 5. Build and score frozen raw evidence

```bash
python scripts/v25_sv_materialize_raw_evidence.py \
  --config output/v25_sv/RESOLVED_CONFIG.json \
  --output-dir output/v25_sv

python scripts/v25_sv_score_raw_evidence.py \
  --config output/v25_sv/RESOLVED_CONFIG.json \
  --block-manifest output/v25_sv/v25_sv_raw_block_manifest.jsonl \
  --block-provenance output/v25_sv/v25_sv_raw_block_manifest.provenance.json \
  --output-dir output/v25_sv \
  --dry-run

python scripts/v25_sv_score_raw_evidence.py \
  --config output/v25_sv/RESOLVED_CONFIG.json \
  --block-manifest output/v25_sv/v25_sv_raw_block_manifest.jsonl \
  --block-provenance output/v25_sv/v25_sv_raw_block_manifest.provenance.json \
  --output-dir output/v25_sv
```

The raw scorer must finish and emit document-pack provenance before the next
stage. Do not reuse incomplete `v25_sv_document_packs.jsonl.tmp` output.

## 6. Verifier preflight and confirmatory OOF

```bash
python scripts/v25_sv_train_oof.py \
  --config output/v25_sv/RESOLVED_CONFIG.json \
  --packs output/v25_sv/v25_sv_document_packs.jsonl \
  --pack-provenance output/v25_sv/v25_sv_document_packs.provenance.json \
  --output-dir output/v25_sv \
  --preflight

python scripts/v25_sv_train_oof.py \
  --config output/v25_sv/RESOLVED_CONFIG.json \
  --packs output/v25_sv/v25_sv_document_packs.jsonl \
  --pack-provenance output/v25_sv/v25_sv_document_packs.provenance.json \
  --output-dir output/v25_sv \
  --oof
```

Read `output/v25_sv/v25_sv_oof_report.json`. The bundle intentionally does
not contain a holdout runner: only the dataset owner may perform the single
holdout evaluation, and only after every preregistered OOF gate passes.
