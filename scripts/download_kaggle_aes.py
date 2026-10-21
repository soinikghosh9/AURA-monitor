#!/usr/bin/env python
"""Resumable, space-safe downloader for the Kaggle AES (American Epilepsy Society) competition.

The competition stores ~8000 individual files (4067 labeled interictal/preictal + 3935 test), each
served as a small per-file `.zip`. The monolithic `kaggle competitions download` would need ~90 GB
peak (zip + extract), so instead this downloads the **labeled files we are missing**, one at a time,
extracting each `.mat` directly and deleting the zip — peak overhead is one file.

Resumable: re-running skips files already present at the right size, so a power cut / Ctrl-C just
continues. The pipeline only uses interictal+preictal, so test segments are skipped by default.

Auth: reads the Kaggle token from ~/.kaggle/access_token (the KGAT_ token) or KAGGLE_API_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

COMP = "seizure-prediction"
DEFAULT_DEST = "datasets/kaggle_aes"
SUBJECTS = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api_connect():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def list_labeled(api, include_test: bool) -> list[tuple[str, int]]:
    """Walk all pages; return [(nested_name, total_bytes)] for interictal/preictal (+test if asked)."""
    out: list[tuple[str, int]] = []
    tok = None
    pages = 0
    while True:
        resp = api.competition_list_files(COMP, page_token=tok, page_size=1000)
        for f in resp.files:
            n = f.name
            keep = ("interictal_segment" in n or "preictal_segment" in n) or (include_test and "test_segment" in n)
            if keep:
                out.append((n, int(getattr(f, "total_bytes", 0) or 0)))
        pages += 1
        tok = resp.next_page_token
        if not tok or len(resp.files) == 0:
            break
    log(f"listed {len(out)} target files across {pages} pages")
    return out


def download_one(api, nested: str, dest: Path, tmp: Path) -> None:
    """Download one competition file (.zip), extract its .mat directly to dest/<basename>."""
    base = os.path.basename(nested)                  # e.g. Dog_5_interictal_segment_0001.mat
    final = dest / base
    api.competition_download_file(COMP, nested, path=str(tmp), force=True, quiet=True)
    zpath = tmp / (base + ".zip")
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            mats = [m for m in z.namelist() if m.endswith(".mat")]
            if not mats:
                raise ValueError(f"no .mat inside {zpath.name}")
            with z.open(mats[0]) as src, open(final, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zpath.unlink()
    elif (tmp / base).exists():                       # served unzipped
        shutil.move(str(tmp / base), str(final))
    else:
        raise FileNotFoundError(f"nothing downloaded for {nested}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--subjects", default="all", help="'all' or comma list e.g. Dog_5,Patient_1")
    ap.add_argument("--include-test", action="store_true", help="also fetch unused test segments")
    ap.add_argument("--limit", type=int, default=0, help="stop after N downloads (0 = no limit; for testing)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.5, help="throttle seconds between files (avoid Kaggle 429)")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_dl_tmp"
    tmp.mkdir(exist_ok=True)

    # Single-instance lock: concurrent downloaders trigger Kaggle 429 rate-limits. Use an ATOMIC
    # O_EXCL create so two processes starting in the same instant can't both pass (the harness may
    # duplicate-spawn a background command). A stale lock (>120 s untouched) is reclaimed.
    lock = tmp / ".lock"

    def _take_lock() -> bool:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    if not _take_lock():
        if lock.exists() and (time.time() - lock.stat().st_mtime) > 120:
            lock.unlink(missing_ok=True)  # stale; reclaim
        if not _take_lock():
            log("another downloader holds the lock; exiting (avoids 429 collisions).")
            return 3
    subjects = SUBJECTS if args.subjects == "all" else [s.strip() for s in args.subjects.split(",")]

    log(f"AES downloader -> {dest}")
    log(f"subjects: {', '.join(subjects)}")
    api = api_connect()
    targets = list_labeled(api, args.include_test)
    # filter by subject
    targets = [(n, sz) for (n, sz) in targets if any(re.search(rf"{s}_", os.path.basename(n)) for s in subjects)]

    todo = []
    have = 0
    for nested, sz in targets:
        final = dest / os.path.basename(nested)
        if final.exists() and (sz == 0 or abs(final.stat().st_size - sz) <= 2):
            have += 1
        else:
            todo.append((nested, sz))
    need_gb = sum(sz for _, sz in todo) / 1e9
    log(f"{len(targets)} labeled files | already have {have} | to download {len(todo)} (~{need_gb:.1f} GB)")
    if args.limit:
        todo = todo[: args.limit]
        log(f"--limit {args.limit}: only fetching {len(todo)} this run")

    done = failed = 0
    bytes_got = 0
    consec_429 = 0
    t0 = time.time()
    try:
        for i, (nested, sz) in enumerate(todo, 1):
            lock.write_text(str(os.getpid()))  # heartbeat
            for attempt in range(1, args.retries + 1):
                try:
                    download_one(api, nested, dest, tmp)
                    done += 1
                    bytes_got += sz
                    consec_429 = 0
                    break
                except KeyboardInterrupt:
                    log("interrupted — re-run to resume (downloaded files are kept).")
                    return 130
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    is_429 = "429" in msg or "Too Many Requests" in msg
                    if attempt == args.retries:
                        failed += 1
                        log(f"  FAILED {os.path.basename(nested)}: {msg[:90]}")
                    elif is_429:
                        consec_429 += 1
                        wait = min(20 * attempt + 15 * consec_429, 300)
                        log(f"  429 rate-limited; backing off {wait}s ({os.path.basename(nested)})")
                        time.sleep(wait)
                    else:
                        time.sleep(2 * attempt)
            # adaptive throttle: base + extra if we've been rate-limited recently
            time.sleep(args.sleep + min(consec_429, 6) * 0.5)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                rate = bytes_got / 1e6 / max(1e-6, el)
                eta = (len(todo) - i) * el / max(1, i) / 60
                log(f"  {i}/{len(todo)}  ({bytes_got/1e9:.1f} GB, {rate:.1f} MB/s, ETA {eta:.0f} min, failed {failed})")
    finally:
        lock.unlink(missing_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    log(f"DONE: downloaded {done}, skipped {have}, failed {failed} in {(time.time()-t0)/60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
