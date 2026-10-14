import csv
import json
import os

import requests

GDC_CASES = "https://api.gdc.cancer.gov/cases"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
CLINICAL_DIR = os.path.join(BASE_DIR, "clinical")

FIELDS = [
    "case_id",
    "submitter_id",
    "demographic.gender",
    "demographic.age_at_index",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    "diagnoses.vascular_invasion_present",
    "diagnoses.primary_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "exposures.alcohol_history",
    "exposures.tobacco_smoking_status",
]


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        case_ids = [r["case_id"] for r in csv.DictReader(f)]

    # POST, not GET: a few hundred case_ids makes the querystring long enough
    # that GDC's API returns 414 Request-URI Too Long on a GET request.
    filters = {"op": "in", "content": {"field": "case_id", "value": case_ids}}
    payload = {"filters": filters, "fields": ",".join(FIELDS), "format": "JSON", "size": len(case_ids)}
    r = requests.post(GDC_CASES, json=payload, timeout=60)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    print(f"fetched clinical data for {len(hits)} cases")

    os.makedirs(CLINICAL_DIR, exist_ok=True)
    raw_path = os.path.join(CLINICAL_DIR, "pilot_clinical_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(hits, f, ensure_ascii=False, indent=2)
    print(f"wrote raw json: {raw_path}")

    rows = []
    for h in hits:
        diag = (h.get("diagnoses") or [{}])[0]
        demo = h.get("demographic") or {}
        expo = (h.get("exposures") or [{}])[0]
        rows.append(
            {
                "case_id": h.get("case_id"),
                "submitter_id": h.get("submitter_id"),
                "gender": demo.get("gender"),
                "age_at_index": demo.get("age_at_index"),
                "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                "ajcc_pathologic_stage": diag.get("ajcc_pathologic_stage"),
                "ajcc_pathologic_t": diag.get("ajcc_pathologic_t"),
                "ajcc_pathologic_n": diag.get("ajcc_pathologic_n"),
                "ajcc_pathologic_m": diag.get("ajcc_pathologic_m"),
                "tumor_grade": diag.get("tumor_grade"),
                "vascular_invasion_present": diag.get("vascular_invasion_present"),
                "primary_diagnosis": diag.get("primary_diagnosis"),
                "days_to_last_follow_up": diag.get("days_to_last_follow_up"),
                "alcohol_history": expo.get("alcohol_history"),
                "tobacco_smoking_status": expo.get("tobacco_smoking_status"),
            }
        )

    csv_path = os.path.join(CLINICAL_DIR, "pilot_clinical.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote clinical csv: {csv_path}")


if __name__ == "__main__":
    main()
