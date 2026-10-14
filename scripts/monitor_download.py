"""Live progress display for 02_download_pilot.py. Run in a separate terminal
while the download runs; refreshes every few seconds until Ctrl+C."""

import csv
import os
import time

import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
WSI_DIR = os.path.join(BASE_DIR, "wsi")
RNASEQ_DIR = os.path.join(BASE_DIR, "rnaseq")
REFRESH_SECONDS = 5


def get_expected_sizes(file_ids):
    filters = {"op": "in", "content": {"field": "file_id", "value": file_ids}}
    payload = {"filters": filters, "fields": "file_id,file_size", "size": len(file_ids)}
    r = requests.post(GDC_FILES, json=payload, timeout=60)
    r.raise_for_status()
    return {h["file_id"]: h["file_size"] for h in r.json()["data"]["hits"]}


def current_bytes_and_counts(files):
    """files: list of (dir, filename, expected_size). Returns (bytes_done, n_complete)."""
    bytes_done, n_complete = 0, 0
    for folder, name, expected in files:
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        bytes_done += min(size, expected)  # a re-downloading (truncated) file counts partial progress
        if size == expected:
            n_complete += 1
    return bytes_done, n_complete


def format_gb(b):
    return f"{b / 1e9:.1f}GB"


def format_eta(seconds):
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # nan/inf
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    file_ids = [r["slide_file_id"] for r in rows] + [r["rnaseq_file_id"] for r in rows]
    print(f"fetching expected file sizes for {len(rows)} cases...")
    sizes = get_expected_sizes(file_ids)

    files = [(WSI_DIR, r["slide_file_name"], sizes.get(r["slide_file_id"], 0)) for r in rows]
    files += [(RNASEQ_DIR, r["rnaseq_file_name"], sizes.get(r["rnaseq_file_id"], 0)) for r in rows]
    total_bytes = sum(expected for _, _, expected in files)
    total_files = len(files)

    prev_bytes, prev_time = None, None
    try:
        while True:
            bytes_done, n_complete = current_bytes_and_counts(files)
            pct = 100 * bytes_done / total_bytes if total_bytes else 0
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)

            now = time.time()
            if prev_bytes is not None and now > prev_time:
                rate = (bytes_done - prev_bytes) / (now - prev_time)
            else:
                rate = 0
            eta = (total_bytes - bytes_done) / rate if rate > 0 else float("nan")

            print(f"\r[{bar}] {pct:5.1f}%  {n_complete}/{total_files} files  "
                  f"{format_gb(bytes_done)}/{format_gb(total_bytes)}  "
                  f"{rate / 1e6:5.1f}MB/s  ETA {format_eta(eta)}   ", end="", flush=True)

            if n_complete == total_files:
                print("\ndone.")
                break

            prev_bytes, prev_time = bytes_done, now
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
