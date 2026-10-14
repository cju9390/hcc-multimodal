import csv
import json
import os

import openpyxl
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
PROJECT = "TCGA-LIHC"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PILOT_SIZE = 185  # = size of the Table S1A core cohort (has an iCluster label) with both WSI+RNA-seq in GDC

# Ally et al., Cell 2017 Table S1A core cohort (see 04_build_metabolic_labels.py) --
# these are the only patients with a published iCluster metabolic-subtype label,
# so they're prioritized first when the pilot is smaller than the full project.
CORE_COHORT_XLSX = os.path.join(OUT_DIR, "external", "NIHMS882929-supplement-8.xlsx")


def load_core_cohort_submitter_ids():
    wb = openpyxl.load_workbook(CORE_COHORT_XLSX, read_only=True, data_only=True)
    ws = wb["Table S1A - core sample set"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[3]
    barcode_idx = header.index("Barcode")
    return {"-".join(r[barcode_idx].split("-")[:3]) for r in rows[4:] if r[barcode_idx]}


def query_files(data_type, extra_filters=None, size=3000):
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
    ]
    if extra_filters:
        content.extend(extra_filters)
    filters = {"op": "and", "content": content}
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,data_type,experimental_strategy,"
        "cases.case_id,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": size,
    }
    r = requests.get(GDC_FILES, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def group_by_case(hits):
    grouped = {}
    for h in hits:
        case = h["cases"][0]
        grouped.setdefault(case["case_id"], []).append(h)
    return grouped


def main():
    slide_hits = query_files(
        "Slide Image",
        extra_filters=[
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}
        ],
    )
    rnaseq_hits = query_files(
        "Gene Expression Quantification",
        extra_filters=[
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}}
        ],
    )

    slide_by_case = group_by_case(slide_hits)
    rna_by_case = group_by_case(rnaseq_hits)

    common_cases = sorted(set(slide_by_case) & set(rna_by_case))
    print(f"slide cases: {len(slide_by_case)}, rnaseq cases: {len(rna_by_case)}, common: {len(common_cases)}")

    # prioritize primary tumor samples for the pilot
    def is_tumor(hit):
        samples = hit["cases"][0].get("samples", [])
        return any(s.get("sample_type") == "Primary Tumor" for s in samples)

    tumor_cases = [c for c in common_cases if is_tumor(slide_by_case[c][0])]
    normal_cases = [c for c in common_cases if c not in tumor_cases]
    print(f"tumor cases: {len(tumor_cases)}, normal/other cases: {len(normal_cases)}")

    core_cohort_ids = load_core_cohort_submitter_ids()

    def submitter_id(case_id):
        return slide_by_case[case_id][0]["cases"][0]["submitter_id"]

    in_core = [c for c in tumor_cases if submitter_id(c) in core_cohort_ids]
    not_in_core = [c for c in tumor_cases if submitter_id(c) not in core_cohort_ids]
    print(f"tumor cases in Table S1A core cohort (has iCluster label): {len(in_core)}")

    pilot = (in_core + not_in_core)[:PILOT_SIZE]

    rows = []
    for case_id in pilot:
        slide = slide_by_case[case_id][0]
        rna = rna_by_case[case_id][0]
        case_info = slide["cases"][0]
        sample_type = ""
        if case_info.get("samples"):
            sample_type = case_info["samples"][0].get("sample_type", "")
        rows.append(
            {
                "case_id": case_id,
                "submitter_id": case_info["submitter_id"],
                "sample_type": sample_type,
                "slide_file_id": slide["file_id"],
                "slide_file_name": slide["file_name"],
                "rnaseq_file_id": rna["file_id"],
                "rnaseq_file_name": rna["file_name"],
            }
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "pilot_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
