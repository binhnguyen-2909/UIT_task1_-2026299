#!/usr/bin/env python3
"""Chunked resumable downloader for Qwen3-1.7B (2000 packets total).

Splits each of the 2 large safetensors shards into 1000 Range chunks
(2 x 1000 = 2000 packets), downloads in parallel with resume bitmap.
Small files are fetched directly. Target: project-local models/Qwen3-1.7B/
so the bench can run with --qwen-model models/Qwen3-1.7B offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "Qwen/Qwen3-1.7B"
LARGE_FILES = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
SMALL_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "README.md", "LICENSE", ".gitattributes",
]

UA = {"User-Agent": "UIT-DSC-chunked-downloader/1.0"}


def _url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        size = resp.getheader("Content-Length")
        if size is None:
            raise RuntimeError(f"no Content-Length for {url}")
        accept = resp.getheader("Accept-Ranges", "")
        if "bytes" not in accept:
            print(f"WARN: server Accept-Ranges={accept!r} for {url}", flush=True)
        return int(size)


def _fetch_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status}")
                data = resp.read()
            expect = end - start + 1
            if len(data) != expect:
                raise RuntimeError(f"short read {len(data)} != {expect}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry any transient error
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def download_large(url: str, dest: Path, chunks: int, workers: int) -> None:
    size = _remote_size(url)
    part = dest.with_suffix(dest.suffix + ".part")
    bitmap_path = dest.with_suffix(dest.suffix + ".bitmap.json")
    if dest.is_file() and dest.stat().st_size == size:
        print(f"SKIP done {dest.name} ({size} bytes)", flush=True)
        return
    # Preallocate part file.
    if not part.is_file() or part.stat().st_size != size:
        print(f"ALLOC {dest.name} {size} bytes in {chunks} chunks", flush=True)
        with open(part, "wb") as fh:
            fh.truncate(size)
        bitmap: dict[str, bool] = {}
    else:
        try:
            bitmap = json.loads(bitmap_path.read_text())
        except OSError:
            bitmap = {}
    chunk_size = (size + chunks - 1) // chunks
    todo = [i for i in range(chunks) if not bitmap.get(str(i))]
    print(f"RESUME {dest.name}: {len(todo)}/{chunks} chunks left", flush=True)
    fd = os.open(part, os.O_WRONLY)

    def one(idx: int) -> int:
        start = idx * chunk_size
        end = min(start + chunk_size - 1, size - 1)
        data = _fetch_range(url, start, end)
        os.pwrite(fd, data, start)
        bitmap[str(idx)] = True
        return idx

    done_count = chunks - len(todo)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx in pool.map(one, todo):
                done_count += 1
                if done_count % 50 == 0 or done_count == chunks:
                    bitmap_path.write_text(json.dumps(bitmap))
                    pct = 100.0 * done_count / chunks
                    print(f"  {dest.name}: {done_count}/{chunks} ({pct:.1f}%)", flush=True)
    finally:
        os.close(fd)
    bitmap_path.write_text(json.dumps(bitmap))
    os.replace(part, dest)
    try:
        bitmap_path.unlink()
    except OSError:
        pass
    print(f"DONE {dest.name} ({dest.stat().st_size} bytes)", flush=True)


def download_small(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"SKIP done {dest.name}", flush=True)
        return
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    print(f"DONE {dest.name} ({dest.stat().st_size} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--dest", type=Path, default=Path("models/Qwen3-1.7B"))
    parser.add_argument("--chunks-per-file", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--large-only", action="store_true")
    args = parser.parse_args()

    dest_dir: Path = args.dest if args.dest.is_absolute() else Path.cwd() / args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_packets = 0
    for fname in LARGE_FILES:
        download_large(
            _url(args.repo, fname), dest_dir / fname,
            chunks=int(args.chunks_per_file), workers=int(args.workers),
        )
        total_packets += int(args.chunks_per_file)
    if not bool(args.large_only):
        for fname in SMALL_FILES:
            try:
                download_small(_url(args.repo, fname), dest_dir / fname)
            except Exception as exc:  # noqa: BLE001 - small files optional
                print(f"WARN small {fname}: {exc}", flush=True)
    print(f"ALL DONE: {total_packets} large-file packets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
