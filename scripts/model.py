"""Dataset and model definitions shared by 07_train_model.py.

Pipeline-validation scope: the goal at this stage is to prove the data ->
tiles -> MIL aggregation -> clinical fusion -> multi-task loss path runs
correctly end to end, not to reach real accuracy (train/val/test are 18/6/6
patients, with subtype labels on barely half of them).
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_PATH = os.path.join(PROCESSED_DIR, "pilot_split.csv")
TILE_MANIFEST_PATH = os.path.join(PROCESSED_DIR, "tile_manifest.csv")
CLINICAL_PATH = os.path.join(DATA_DIR, "raw", "clinical", "pilot_clinical.csv")

SUBTYPE_TO_IDX = {"iCluster1": 0, "iCluster2": 1, "iCluster3": 2}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def stage_to_label(ajcc_stage):
    if not ajcc_stage:
        return None
    if "III" in ajcc_stage or "IV" in ajcc_stage:
        return 1  # late
    if "I" in ajcc_stage or "II" in ajcc_stage:
        return 0  # early
    return None


class ClinicalFeaturizer:
    """age_at_index (normalized) + vital_status (Dead=1/Alive=0). Every other
    clinical field is empty for this pilot cohort (see README), so this is
    deliberately the minimal feature set rather than something padded out."""

    def fit(self, rows):
        ages = [float(r["age_at_index"]) for r in rows if r["age_at_index"]]
        self.age_mean = float(np.mean(ages))
        self.age_std = float(np.std(ages)) or 1.0
        return self

    def transform(self, row):
        age = float(row["age_at_index"]) if row["age_at_index"] else self.age_mean
        age_norm = (age - self.age_mean) / self.age_std
        vital = 1.0 if row["vital_status"] == "Dead" else 0.0
        return torch.tensor([age_norm, vital], dtype=torch.float32)


def load_patient_table():
    with open(SPLIT_PATH, newline="", encoding="utf-8") as f:
        split_rows = {r["submitter_id"]: r for r in csv.DictReader(f)}

    with open(CLINICAL_PATH, newline="", encoding="utf-8") as f:
        clinical_rows = {r["submitter_id"]: r for r in csv.DictReader(f)}

    with open(TILE_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        tiles_by_patient = {}
        for r in csv.DictReader(f):
            tiles_by_patient.setdefault(r["submitter_id"], []).append(r["tile_path"])

    patients = []
    for submitter_id, split_row in split_rows.items():
        clinical = clinical_rows[submitter_id]
        patients.append(
            {
                "submitter_id": submitter_id,
                "split": split_row["split"],
                "subtype_label": SUBTYPE_TO_IDX.get(split_row["metabolic_subtype"]),
                "stage_label": stage_to_label(clinical["ajcc_pathologic_stage"]),
                "clinical_row": clinical,
                "tile_paths": tiles_by_patient[submitter_id],
            }
        )
    return patients


class WSIBagDataset(Dataset):
    def __init__(self, patients, featurizer):
        self.patients = patients
        self.featurizer = featurizer
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        bag = torch.stack(
            [self.transform(Image.open(os.path.join(PROCESSED_DIR, path)).convert("RGB")) for path in p["tile_paths"]]
        )
        clinical = self.featurizer.transform(p["clinical_row"])
        stage_label = p["stage_label"] if p["stage_label"] is not None else -1
        subtype_label = p["subtype_label"] if p["subtype_label"] is not None else -1
        return bag, clinical, stage_label, subtype_label, p["submitter_id"]


class AttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al., 2018) over per-tile features."""

    def __init__(self, feat_dim=512, hidden_dim=128):
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, tile_feats):
        # tile_feats: (N, feat_dim) -> bag_feat: (feat_dim,)
        gate = self.attn_w(self.attn_v(tile_feats) * self.attn_u(tile_feats))  # (N, 1)
        weights = torch.softmax(gate, dim=0)
        return (weights * tile_feats).sum(dim=0), weights.squeeze(-1)


class MultiTaskHCCModel(nn.Module):
    def __init__(self, clinical_dim=2, n_subtypes=3):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # drop the classifier head
        self.mil = AttentionMIL(feat_dim=512)
        self.fusion = nn.Sequential(nn.Linear(512 + clinical_dim, 128), nn.ReLU())
        self.stage_head = nn.Linear(128, 1)
        self.subtype_head = nn.Linear(128, n_subtypes)

    def forward(self, bag, clinical, encode_chunk=32):
        # bag: (N, 3, H, W). Encode in chunks so a 100-tile bag doesn't blow up
        # activation memory on a 6GB GPU the way one N=100 forward pass would.
        feats = []
        for i in range(0, bag.shape[0], encode_chunk):
            chunk = self.encoder(bag[i : i + encode_chunk]).flatten(1)
            feats.append(chunk)
        tile_feats = torch.cat(feats, dim=0)

        bag_feat, attn_weights = self.mil(tile_feats)
        fused = self.fusion(torch.cat([bag_feat, clinical], dim=0))
        stage_logit = self.stage_head(fused).squeeze(-1)
        subtype_logits = self.subtype_head(fused)
        return stage_logit, subtype_logits, attn_weights
