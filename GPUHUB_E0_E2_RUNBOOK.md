# LegalIR E0-E2 GPUHub capsule

## Nhanh nhất: chạy tự động

Với bản `LegalIR_E0_E2_Server_Auto.zip`, xem `SERVER_AUTO_RUN.md` và chạy
`bash ./run_e0_e2_auto.sh` trong `screen`. Launcher tự cài dependency (trừ
PyTorch/CUDA đã có trong GPU image), đọc VRAM, chạy E2 exact memory probe,
và chỉ chạy các E2 stress gate còn thiếu. Nó không chạy lại standard OOF.

This package contains the reproducible LegalIR baseline and the research
assets for the proposed E0 -> E1 -> E2 direction:

- the complete `Task1 - LegalIR` data, code, audits, splits and V16b artifacts;
- the BQB Vietnamese embedding model;
- the Vietnamese reranker used by the current pipeline;
- a Linux launcher for the locked V16b baseline;
- the dependency and GPUHub instructions needed to continue the research.

The launcher first reproduces locked V16b. The new E1/E2 research direction is
not treated as already validated: its experiments must be run and evaluated
against this baseline under the included split/audit protocol.

## GPUHub setup

Use the data disk, not `/`, for this package. The expected extraction root is
`/root/autodl-tmp/legalir`.

```bash
mkdir -p /root/autodl-tmp/legalir
unzip -q "LegalIR_E0_E2_GPUHub.zip" -d /root/autodl-tmp/legalir
cd /root/autodl-tmp/legalir
```

Install the runtime dependencies. The PyTorch image already supplies the
CUDA-enabled `torch` package, so the bundle requirements do not reinstall it:

```bash
python3 -m pip install --no-cache-dir -r requirements-gpuhub.txt
```

Verify CUDA before starting a long run:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Reproduce locked V16b

Run inside `screen` so an SSH disconnect does not stop the job:

```bash
screen -S legalir
cd /root/autodl-tmp/legalir
bash ./run_v16b_gpu.sh
```

Detach with `Ctrl+A`, then `D`. Reattach with `screen -r legalir`.

The main result is written under:

```text
Task1 - LegalIR/output/v16_rgm/v16b_memory_ltr_report.json
Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json
```

The V16b runner requires the fit-only prior index and the frozen memory/audit
artifacts included in this capsule. Do not remove
`data/processed/v11_research_fit_256/legalir.sqlite3` if you want to reproduce
the locked result.

## Continue E0 -> E1 -> E2

Use the locked V16b result as the baseline. Keep the included split files,
provenance, audit outputs and cache metadata. Any new candidate pool or
passage selector must regenerate downstream scores; do not reuse old scores
after changing the passage pack.

The package includes the existing experiment scripts and documents under:

```text
Task1 - LegalIR/experiments/track_b_v11_research/
Task1 - LegalIR/scripts/
```

Do not use public labels to choose thresholds or policies. Promote a change
only after it wins the locked evaluation under the registered protocol.
