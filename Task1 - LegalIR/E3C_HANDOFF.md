# E3-C server handoff

Run this only after E3-B’s retained artifacts have been copied intact:

- `output/e3b_nested_mil/e3b_standard_report.json`
- `output/e3b_nested_mil/e3b_stress_report.json`
- `output/e3b_nested_mil/e3b_weights/**` (all 40 retained source weights)
- `output/e3b_nested_mil/e3b_full_scores/**` (all C/V exports and metadata)

E3-C deliberately fails preflight when any of those are missing. It must not
fall back to the base model. Install the added dependency once:

```bash
pip install -r requirements-gpuhub.txt
```

First run the CUDA runtime-profile eligibility check. It does not train or
open holdout, and all numerical CPU worker pools are capped at one thread:

```bash
cd /root/autodl-tmp/legalir
CPU_THREADS=1 MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3c_nested_duo_lora.sh" \
  2>&1 | tee /root/autodl-tmp/e3c_preflight.log
```

Only if it prints an `e3c_preflight.json` with `holdout_opened:false`, run the
10-fit standard experiment. Use screen or tmux because it is a GPU job:

```bash
screen -S e3c
cd /root/autodl-tmp/legalir
MODE=standard bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3c_nested_duo_lora.sh" \
  2>&1 | tee /root/autodl-tmp/e3c_standard.log
```

After completion, inspect `e3c_standard_report.json`. Run stress only if
`promotion_to_stress_allowed` is true. The runner resumes completed outer
folds and preserves per-fold optimizer/adapter checkpoints:

```bash
MODE=stress bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3c_nested_duo_lora.sh" \
  2>&1 | tee /root/autodl-tmp/e3c_stress.log
```

Do not run E4 because of E3-C. E4 remains the independent one-time parity
validation of locked E3-B.
