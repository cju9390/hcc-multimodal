import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ClinicalFeaturizer, MultiTaskHCCModel, WSIBagDataset, load_patient_table

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "checkpoints")


def split_loss(stage_logit, subtype_logits, stage_label, subtype_label):
    """Multi-task loss for one patient, skipping a head when its label is
    missing (-1) -- about half the pilot cohort lacks a subtype label and
    ~30% lacks a stage label, so most patients only supervise one head."""
    loss = torch.tensor(0.0, device=stage_logit.device)
    n_terms = 0
    if stage_label.item() != -1:
        loss = loss + nn.functional.binary_cross_entropy_with_logits(stage_logit.view(-1), stage_label.view(-1).float())
        n_terms += 1
    if subtype_label.item() != -1:
        loss = loss + nn.functional.cross_entropy(subtype_logits.unsqueeze(0), subtype_label)
        n_terms += 1
    return loss, n_terms


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    stage_correct = stage_total = 0
    subtype_correct = subtype_total = 0
    for bag, clinical, stage_label, subtype_label, submitter_id in loader:
        bag, clinical = bag[0].to(device), clinical[0].to(device)
        stage_label, subtype_label = stage_label.to(device), subtype_label.to(device)

        stage_logit, subtype_logits, _ = model(bag, clinical)

        if stage_label.item() != -1:
            pred = int(torch.sigmoid(stage_logit).item() > 0.5)
            stage_correct += int(pred == stage_label.item())
            stage_total += 1
        if subtype_label.item() != -1:
            pred = int(subtype_logits.argmax().item())
            subtype_correct += int(pred == subtype_label.item())
            subtype_total += 1

    stage_acc = stage_correct / stage_total if stage_total else float("nan")
    subtype_acc = subtype_correct / subtype_total if subtype_total else float("nan")
    return stage_acc, stage_total, subtype_acc, subtype_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    patients = load_patient_table()
    by_split = {"train": [], "val": [], "test": []}
    for p in patients:
        by_split[p["split"]].append(p)

    featurizer = ClinicalFeaturizer().fit([p["clinical_row"] for p in by_split["train"]])

    loaders = {
        split: DataLoader(WSIBagDataset(rows, featurizer), batch_size=1, shuffle=(split == "train"))
        for split, rows in by_split.items()
    }
    print(f"patients: train={len(by_split['train'])} val={len(by_split['val'])} test={len(by_split['test'])}")

    model = MultiTaskHCCModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_patients_with_loss = 0.0, 0
        for bag, clinical, stage_label, subtype_label, submitter_id in loaders["train"]:
            bag, clinical = bag[0].to(device), clinical[0].to(device)
            stage_label, subtype_label = stage_label.to(device), subtype_label.to(device)

            stage_logit, subtype_logits, _ = model(bag, clinical)
            loss, n_terms = split_loss(stage_logit, subtype_logits, stage_label, subtype_label)
            if n_terms == 0:
                continue  # shouldn't happen (every patient has >=1 label), but be safe

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_patients_with_loss += 1

        avg_loss = total_loss / n_patients_with_loss
        stage_acc, n_stage, subtype_acc, n_subtype = evaluate(model, loaders["val"], device)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={avg_loss:.4f}  "
            f"val_stage_acc={stage_acc:.2f} (n={n_stage})  val_subtype_acc={subtype_acc:.2f} (n={n_subtype})"
        )

    stage_acc, n_stage, subtype_acc, n_subtype = evaluate(model, loaders["test"], device)
    print(f"test_stage_acc={stage_acc:.2f} (n={n_stage})  test_subtype_acc={subtype_acc:.2f} (n={n_subtype})")

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "multitask_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
