# LegalIR E1/E2 bundle — RTX 5070 desktop

This bundle preserves the completed E1 artifacts and E2 standard OOF report,
then provides the only unrun E2 gates: law/source, query-template and
near-duplicate stress OOF.  It does **not** rerun E1 or E2 standard.

The desktop GeForce RTX 5070 has 12GB VRAM.  E2 full fine-tunes a 760M
cross-encoder in FP32 with AdamW.  The launcher therefore uses micro-batch 1,
score batch 1 and activation checkpointing, but these settings may still not
fit the optimizer state.  Do not start the long stress run until the probe
passes.

## Install

Create a Python environment that has a CUDA build of PyTorch compatible with
your RTX 5070, then install the rest:

```powershell
python -m pip install -r requirements-rtx5070.txt
```

## Verify real CUDA memory

From the bundle root:

```powershell
nvidia-smi
& ".\Task1 - LegalIR\experiments\track_b_v11_research\run_e2_stress_rtx5070.ps1" -Mode probe
```

The probe writes `Task1 - LegalIR/output/e2_mil_oof/e2_mil_oof_memory_probe.json`.
If it reports CUDA OOM, this desktop cannot run the exact full-AdamW E2
protocol.  Keep the sealed E1/E2 artifacts and use a GPU with at least 24GB
for stress OOF; do not substitute a different optimizer under this protocol.

## Run the remaining stress gates

Only after a successful probe:

```powershell
& ".\Task1 - LegalIR\experiments\track_b_v11_research\run_e2_stress_rtx5070.ps1" -Mode stress
```

After each completed fold, the launcher atomically updates
`Task1 - LegalIR/output/e2_mil_oof/e2_mil_oof_stress_progress.json`.  Rerun
the same command to resume; use `-Fresh` only to intentionally discard that
stress checkpoint.
