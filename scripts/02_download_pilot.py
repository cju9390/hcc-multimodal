import csv
import os
import time

import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
WSI_DIR = os.path.join(BASE_DIR, "wsi")
RNASEQ_DIR = os.path.join(BASE_DIR, "rnaseq")
MAX_RETRIES = 5


def get_file_sizes(file_ids):
    # POST, not GET: a few hundred file_ids makes the querystring long enough
    # that GDC's API silently chokes on a GET request.
    filters = {"op": "in", "content": {"field": "file_id", "value": file_ids}}
    payload = {"filters": filters, "fields": "file_id,file_size", "size": len(file_ids)}
    r = requests.post(GDC_FILES, json=payload, timeout=60)
    r.raise_for_status()
    return {h["file_id"]: h["file_size"] for h in r.json()["data"]["hits"]}


def download_file(file_id, out_path, expected_size=None):
    if os.path.exists(out_path):
        if expected_size is None or os.path.getsize(out_path) == expected_size:
            print(f"  skip (exists): {os.path.basename(out_path)}")
            return
        print(f"  incomplete, re-downloading: {os.path.basename(out_path)}")

    # A multi-hour, multi-hundred-GB run WILL hit transient network errors
    # (connection resets, timeouts) -- retry with backoff instead of letting
    # one blip kill everything downloaded so far.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(f"{GDC_DATA}/{file_id}", stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            print(f"  done: {os.path.basename(out_path)}")
            return
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if os.path.exists(out_path):
                os.remove(out_path)  # don't leave a truncated file behind
            if attempt == MAX_RETRIES:
                raise
            wait = 5 * attempt
            print(f"  retry {attempt}/{MAX_RETRIES} after error ({e}); waiting {wait}s")
            time.sleep(wait)


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    slide_ids = [r["slide_file_id"] for r in rows]
    rna_ids = [r["rnaseq_file_id"] for r in rows]
    sizes = get_file_sizes(slide_ids + rna_ids)

    total_bytes = sum(sizes.get(fid, 0) for fid in slide_ids + rna_ids)
    print(f"pilot: {len(rows)} cases, estimated total download size: {total_bytes / 1e9:.2f} GB")

    os.makedirs(WSI_DIR, exist_ok=True)
    os.makedirs(RNASEQ_DIR, exist_ok=True)

    failed = []
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {r['submitter_id']}")
        try:
            download_file(
                r["slide_file_id"],
                os.path.join(WSI_DIR, r["slide_file_name"]),
                sizes.get(r["slide_file_id"]),
            )
            download_file(
                r["rnaseq_file_id"],
                os.path.join(RNASEQ_DIR, r["rnaseq_file_name"]),
                sizes.get(r["rnaseq_file_id"]),
            )
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"  FAILED after {MAX_RETRIES} retries: {r['submitter_id']} ({e})")
            failed.append(r["submitter_id"])

    if failed:
        print(f"\n{len(failed)} case(s) failed, re-run this script to retry them: {failed}")
    else:
        print("\nall cases downloaded successfully")


if __name__ == "__main__":
    main()
