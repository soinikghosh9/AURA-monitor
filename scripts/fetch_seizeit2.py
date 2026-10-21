#!/usr/bin/env python
"""Download the SeizeIT2 wearable dataset (OpenNeuro ds005873 v1.1.0, CC0): the whole dataset.

Lists the public OpenNeuro S3 bucket and mirrors every file (behind-the-ear EEG, ECG, EMG and
movement EDFs, JSON sidecars, seizure ``events.tsv`` and the dataset-level metadata) into ``--dest``
with the BIDS layout unchanged. ``--eeg-only`` restricts it to the metadata and the ``eeg/`` folders.
Resumable: a file whose local size equals the listed size is skipped, a partial download is written to
``*.part`` and re-fetched, and a failed file is retried with back-off; re-running the script completes
whatever is missing. Parallel, anonymous HTTPS; no account or credentials.

  python scripts/fetch_seizeit2.py [--dest datasets/seizeit2] [--workers 6] [--eeg-only] [--list-only]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import time
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds005873/"


def list_objects():
    out, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={PREFIX}" + (f"&continuation-token={urllib.request.quote(token)}" if token else "")
        with urllib.request.urlopen(url, timeout=60) as r:
            t = r.read().decode()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", t, flags=re.S)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            return out
        token = m.group(1)


def wanted(key: str, eeg_only: bool) -> bool:
    rel = key[len(PREFIX):]
    if not eeg_only or "/" not in rel:
        return True                                     # everything, or dataset-level metadata
    return "/eeg/" in rel


def fetch(key, size, dest: Path, tries=5):
    p = dest / key[len(PREFIX):]
    if p.exists() and p.stat().st_size == size:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{urllib.request.quote(key)}", timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} != {size}")
            tmp.replace(p)
            return size
        except Exception as e:  # noqa: BLE001
            time.sleep(5 * (k + 1))
            last = e
    raise RuntimeError(f"{key}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="datasets/seizeit2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eeg-only", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest)
    objs = [(k, s) for k, s in list_objects() if wanted(k, a.eeg_only) and not k.endswith("/")]
    total = sum(s for _, s in objs)
    n_edf = sum(k.endswith("_eeg.edf") for k, _ in objs)
    print(f"{len(objs)} files ({n_edf} EEG EDFs), {total / 1e9:.1f} GB to mirror into {dest}", flush=True)
    if a.list_only:
        return 0
    done, t0, failed = 0, time.time(), []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, k, s, dest): k for k, s in sorted(objs, key=lambda x: x[0])}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                done += f.result()
            except Exception as e:  # noqa: BLE001
                failed.append(str(e)[:200])
            if i % 100 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)} files, {done / 1e9:.1f} GB new in {el / 60:.1f} min "
                      f"({done / max(el, 1) / 1e6:.1f} MB/s), {len(failed)} failed", flush=True)
    for e in failed:
        print("FAILED", e, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
