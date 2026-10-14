import csv
import os

import openpyxl

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
SOURCE_XLSX = os.path.join(BASE_DIR, "external", "NIHMS882929-supplement-8.xlsx")
OUT_PATH = os.path.join(BASE_DIR, "clinical", "pilot_metabolic_labels.csv")

# Column names as they appear in "Table S1A - core sample set" (Ally et al.,
# Cell 2017, TCGA-LIHC marker paper, supplementary Table S1A).
COLUMNS = {
    "barcode": "Barcode",
    "icluster": "iCluster clusters (k=3, Ronglai Shen)",
    "hoshida": "Hoshida",
    "hcc_subtype": "HCC subtypes",
    "grade": "neoplasm_histologic_grade",
    "vascular_invasion": "Vascular Invasion",
    "afp": "alpha_fetoprotien_at_procurement",
}


def load_table_s1a():
    wb = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    ws = wb["Table S1A - core sample set"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[3]
    col_idx = {key: header.index(name) for key, name in COLUMNS.items()}

    table = {}
    for r in rows[4:]:
        barcode = r[col_idx["barcode"]]
        if not barcode:
            continue
        submitter_id = "-".join(barcode.split("-")[:3])
        table[submitter_id] = {key: r[idx] for key, idx in col_idx.items() if key != "barcode"}
    return table


def main():
    table_s1a = load_table_s1a()

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        pilot_ids = [row["submitter_id"] for row in csv.DictReader(f)]

    fieldnames = ["submitter_id", "has_label", "metabolic_subtype"] + [
        k for k in COLUMNS if k not in ("barcode", "icluster")
    ]

    matched = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for submitter_id in pilot_ids:
            info = table_s1a.get(submitter_id)
            # A row can exist in Table S1A (barcode present) but still carry the
            # literal string "NA" in the iCluster column -- that case was in the
            # core cohort but excluded from this particular sub-analysis, so it's
            # not a usable label either.
            icluster = (info["icluster"] or "") if info else ""
            has_label = icluster.startswith("iCluster:")

            row = {"submitter_id": submitter_id}
            if has_label:
                matched += 1
                row["has_label"] = True
                row["metabolic_subtype"] = icluster.replace("iCluster:", "iCluster")
                for k in fieldnames[3:]:
                    row[k] = info.get(k, "")
            else:
                row["has_label"] = False
                row["metabolic_subtype"] = ""
                for k in fieldnames[3:]:
                    row[k] = ""
            writer.writerow(row)

    print(f"pilot: {len(pilot_ids)} cases, matched with Table S1A iCluster label: {matched}")
    print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
