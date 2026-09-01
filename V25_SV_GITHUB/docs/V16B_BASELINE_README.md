# V16b 0.9210 release capsule

This directory freezes the only eligible submission lineage at the 0.9210 public snapshot: **V11h retrieval/reranking/fusion** followed by the **V16b QREL-memory gate**. It is a release manifest, not a second copy of the multi-gigabyte indexes or model weights; source artifacts remain in their working locations so the original runners still work.

## Frozen result

| Item | Value |
|---|---|
| Public snapshot | `0.9210` |
| Locked holdout Recall@5 | `0.9228333333` |
| V11h locked anchor | `0.9188333333` |
| Locked delta | `+0.0040000000` |
| Submission | `output/v16b_memory_slot5.zip` |
| ZIP SHA-256 | `B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9` |
| Parameter ledger | BQB 330M + reranker 760M + selector <=10K = 1.09001B (<4B) |

The V16b policy keeps V11h's candidate list and ranks 1--4 byte-for-byte. It may replace only rank 5, using fit-only QREL-memory features, a memory floor of `0.005`, and a non-negative selector margin. It made 177 locked-holdout changes: 9 rescues and 4 destructive changes.

## Use it

1. Run the non-mutating integrity check:

   ```powershell
   Task1 - LegalIR\release\v16b_0921\verify_release.ps1 -VerifyHashes
   ```

2. Submit the frozen file directly: `Task1 - LegalIR/output/v16b_memory_slot5.zip`. Do not use a V18--V20 artifact as a substitute.

3. To rebuild the complete lineage from its inputs, run:

   ```powershell
   Task1 - LegalIR\release\v16b_0921\run_release.ps1 -Device cuda
   ```

   `-Force` recomputes the cacheable V11h/V16 artifacts. It writes the same named artifacts in `Task1 - LegalIR/output/`; first copy the release output if you need to preserve a separate before/after result.

See [MANIFEST.md](MANIFEST.md) for retained paths and [LESSONS_AND_BOTTLENECKS.md](LESSONS_AND_BOTTLENECKS.md) for the evidence behind stopping at this release.
