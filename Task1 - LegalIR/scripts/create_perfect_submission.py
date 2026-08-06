"""Backward-compatible alias for the labelled warmup sanity submission."""

try:
    from .create_baseline import main
except ImportError:
    from create_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
