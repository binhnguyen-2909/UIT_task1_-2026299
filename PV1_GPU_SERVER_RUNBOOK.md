# PV1 GPU Server Runbook

This package is a new, complement-only experiment. It does not replace the
accepted V16b/E4 artifacts, does not rerun E/F, and must not be used to open
holdout.

## Before any GPU command

Copy the pinned **F shortlist plus packed-text** file and its F source
manifest from the E/F server output. They must follow the exact schema in
`Task1 - LegalIR/PV1_PAIRWISE_ABSTAIN_PROTOCOL.md`. Do not invent packed text
from raw corpus: PV1 must compare the exact long F blocks. Also have a local
offline snapshot of `Qwen/Qwen3-1.7B` and record its immutable revision.

Run the following at the repository root. Set paths to real files; do not
replace hashes or use a public-label file.

```bash
cd /root/autodl-tmp/legalir
export F_SHORTLIST="/root/autodl-tmp/frozen_f_shortlist_with_packs.jsonl"
export F_MANIFEST="/root/autodl-tmp/frozen_f_source_manifest.json"
export PV1_MODEL="/root/autodl-tmp/models/Qwen3-1.7B"
export PV1_MODEL_REVISION="<pinned-Hugging-Face-revision-or-snapshot-hash>"
export VRAM_RESERVE_GIB=12
export CPU_THREADS=1
export PV1_RUN_ID=pv1_v1

MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh" \
  2>&1 | tee /root/autodl-tmp/pv1_preflight.log
```

Inspect `Task1 - LegalIR/output/pv1_pairwise_abstain/pv1_v1/pv1_preflight_report.json`.
It must report `under_4b: true`; inspect `pv1_opportunity_audit.json` for the
three separate opportunity definitions. A failed hash/c67/packed-text check is
a data-lineage issue—do not work around it by editing the report.

## GPU stages

Run only after the preflight review:

```bash
MODE=pilot bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh" \
  2>&1 | tee /root/autodl-tmp/pv1_pilot.log

MODE=cache bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh" \
  2>&1 | tee /root/autodl-tmp/pv1_cache.log
```

`pilot` deterministically scores 64 QIDs. `cache` resumes it and scores every
remaining pair. The maximum is 7,182 GPU forward passes total, already
including both document orders. It writes one cache shard per GPU worker; set
distinct `SHARD_INDEX` and a common `SHARD_COUNT` only when workers are given
separate GPU UUIDs. Do not let two workers write the same shard.

## Evaluation

```bash
MODE=eval-preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh"
MODE=standard bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh" \
  2>&1 | tee /root/autodl-tmp/pv1_standard.log
```

Run `MODE=stress` only when `pv1_standard_report.json` has
`promotion_to_stress_allowed: true`. The runner imports the bound existing
gate source and records hashes; it does not silently loosen the gate.

```bash
MODE=stress bash "Task1 - LegalIR/experiments/track_b_v11_research/run_pv1_pairwise_abstain.sh" \
  2>&1 | tee /root/autodl-tmp/pv1_stress.log
```

Do not start public work unless `promotion_to_public_allowed: true` and the
time budget for its separately preflighted F inputs/cache and bundle is known.
`MODE=finalize` refuses to run before a passing stress report, an official
timezone-qualified deadline, a verified fresh-model reload for each cache
shard, and sufficient measured remaining time.
