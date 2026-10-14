import csv
import os
import random

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
LABELS_PATH = os.path.join(BASE_DIR, "clinical", "pilot_metabolic_labels.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pilot_split.csv")

SEED = 42
VAL_FRAC = 0.2
TEST_FRAC = 0.2


def stratified_split(submitter_ids, seed=SEED):
    """Split submitter_ids into train/val/test at the patient level, with
    val/test each getting max(1, round(n*frac)) patients so every group
    (including small ones) is represented in every split."""
    ids = list(submitter_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * TEST_FRAC))
    n_val, n_test = min(n_val, n), min(n_test, n - n_val)
    return {
        "val": ids[:n_val],
        "test": ids[n_val : n_val + n_test],
        "train": ids[n_val + n_test :],
    }


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        pilot_ids = [row["submitter_id"] for row in csv.DictReader(f)]

    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        labels = {row["submitter_id"]: row["metabolic_subtype"] for row in csv.DictReader(f) if row["has_label"] == "True"}

    # Stratify by metabolic subtype where available; patients without a subtype
    # label form their own group so every split still gets full patient coverage
    # (needed for the stage task / image-encoder pretraining, which aren't
    # limited to the 15 subtype-labeled cases).
    groups = {}
    for submitter_id in pilot_ids:
        key = labels.get(submitter_id, "no_label")
        groups.setdefault(key, []).append(submitter_id)

    split_of = {}
    for key, ids in groups.items():
        for split_name, split_ids in stratified_split(ids).items():
            for submitter_id in split_ids:
                split_of[submitter_id] = split_name

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["submitter_id", "split", "metabolic_subtype"])
        writer.writeheader()
        for submitter_id in pilot_ids:
            writer.writerow(
                {
                    "submitter_id": submitter_id,
                    "split": split_of[submitter_id],
                    "metabolic_subtype": labels.get(submitter_id, ""),
                }
            )

    print(f"wrote: {OUT_PATH}")
    for split_name in ("train", "val", "test"):
        ids = [s for s, sp in split_of.items() if sp == split_name]
        counts = {}
        for s in ids:
            key = labels.get(s, "no_label")
            counts[key] = counts.get(key, 0) + 1
        print(f"{split_name}: {len(ids)} patients, by subtype: {counts}")


if __name__ == "__main__":
    main()
