# Private assets to provide outside GitHub

Only share these assets if the BTC rules permit sharing organizer-provided
development data with the intended collaborator. Keep them in the `assets/`
directory described below.

| Destination on server | Source in the local project | Approx. size | Purpose |
| --- | --- | ---: | --- |
| `assets/v18b_metadata.json` | `Task1 - LegalIR/archive/2026-08-31_retired_experiments/Task1-output/v18b_evidence_packs_corrected_r3/v18b_metadata.json` | 3.6 MB | Clean-complement queries, V16b anchor, c35 candidates and qrels. |
| `assets/v23_b0_report.json` | `Task1 - LegalIR/output/v23_b0/v23_b0_report.json` | 0.04 MB | V23-B0 audit contract. |
| `assets/v23_b0_edit_rows.jsonl` | `Task1 - LegalIR/output/v23_b0/v23_b0_edit_rows.jsonl` | 26 MB | Fixed edit beam and qrel-derived labels. |
| `assets/fold_manifest.json` | `Task1 - LegalIR/output/v21_p3_gx/fold_manifest.json` | 0.93 MB | Four audited grouped OOF regimes. |
| `assets/legalir.sqlite3` | `Task1 - LegalIR/data/processed/v11_research_complement_256/legalir.sqlite3` | 947 MB | Raw legal text used to build BGE-Gemma evidence blocks. |

Download the model on the server whenever possible. If that is impossible,
send this private directory separately (about 9.36 GB):

```text
models/bge-reranker-v2-gemma-1787044f/
```

Do not send or upload these files:

- locked holdout or public-label files;
- submission files;
- `v25_sv_document_packs.jsonl.tmp` or other incomplete runs;
- `v25_sv_raw_block_manifest.jsonl` (regenerate it on the server);
- E3B scores/caches: V25 does not use them as model inputs.

