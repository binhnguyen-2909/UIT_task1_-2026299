# Track B — GitHub v10 rebuild

This runner starts from the exact v10 source restored from `origin/main`
(`df086ca2be49cc5d185dfd5a542c1009e1429748`) and uses fixed 1,200-character
chunks with 150-character overlap, BQB dense retrieval and the v10 reranker.

The v10 rebuild is intentionally isolated at `data/processed/v10_github_rebuild`
and never reads the A2/v57 caches.  CPU mode is used when CUDA is unavailable;
the command is resumable through the dense-cache checkpoints produced by the
upstream v10 implementation.
