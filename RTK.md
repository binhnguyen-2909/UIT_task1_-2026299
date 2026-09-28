# RTK - Rust Token Killer (Codex CLI, project-local Windows setup)

**Usage**: Token-optimized CLI proxy for noisy shell commands. The verified
binary is project-local and the wrapper keeps analytics/tee files inside the
writable workspace.

## Rule

From the repository root, route supported noisy commands through the wrapper
by default. This includes file reads, code search, git status/diff,
build/test/lint output, and Docker/Kubernetes logs:

```powershell
& '.\rtk.ps1' git status
& '.\rtk.ps1' rg 'pattern' 'path'
& '.\rtk.ps1' pytest -q
& '.\rtk.ps1' read 'large-log.txt'
```

Do not force PowerShell built-ins, GPU training processes, or unsupported
commands through RTK. For those, write verbose output to an artifact and use
`rtk.ps1 read`, `log`, or `smart` to inspect a compact result. If filtering
hides information needed for diagnosis, fall back to the original command or
read the project-local `.rtk/tee` artifact.

## Meta Commands

```powershell
& '.\rtk.ps1' gain
& '.\rtk.ps1' gain --history
& '.\rtk.ps1' proxy <cmd>
```

## Verification

```powershell
& '.\rtk.ps1' --version
& '.\rtk.ps1' gain
Get-Item '.\.tools\rtk-0.45.0\rtk.exe'
```
