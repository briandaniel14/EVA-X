#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_patient_id(path_value: str) -> str:
    match = re.search(r"(patient\d+)", str(path_value))
    if match is None:
        raise ValueError(f"Could not extract patient id from Path='{path_value}'")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a patient-level train/val split for CheXpert, optionally filtered by view (e.g., lateral-only)."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to CheXpert train.csv (or any CheXpert CSV) to split",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write train.csv and val.csv",
    )
    parser.add_argument(
        "--view",
        default="lateral",
        choices=["lateral", "frontal", "all"],
        help="Which views to keep before splitting (default: lateral)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of patients to put into validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the patient shuffle (default: 0)",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if "Path" not in df.columns:
        raise ValueError("Input CSV must have a 'Path' column")
    if "Frontal/Lateral" not in df.columns:
        raise ValueError("Input CSV must have a 'Frontal/Lateral' column")

    view = args.view.lower().strip()
    if view == "lateral":
        df = df[df["Frontal/Lateral"] == "Lateral"].copy()
    elif view == "frontal":
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    elif view == "all":
        df = df.copy()

    if len(df) == 0:
        raise ValueError(f"No rows left after view filter: view='{view}'")

    df["_PatientID"] = df["Path"].apply(extract_patient_id)

    patient_ids = df["_PatientID"].unique().tolist()
    patient_ids = sorted(patient_ids)

    rng = np.random.RandomState(args.seed)
    rng.shuffle(patient_ids)

    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val_frac must be between 0 and 1")

    num_val = max(1, int(round(len(patient_ids) * args.val_frac)))
    val_patients = set(patient_ids[:num_val])
    train_patients = set(patient_ids[num_val:])

    train_df = df[df["_PatientID"].isin(train_patients)].drop(columns=["_PatientID"]).copy()
    val_df = df[df["_PatientID"].isin(val_patients)].drop(columns=["_PatientID"]).copy()

    # Sanity checks
    train_patient_set = set(train_df["Path"].apply(extract_patient_id))
    val_patient_set = set(val_df["Path"].apply(extract_patient_id))
    overlap = train_patient_set & val_patient_set
    if overlap:
        raise RuntimeError(f"Patient leakage detected! Overlap size: {len(overlap)}")

    train_out = output_dir / "train.csv"
    val_out = output_dir / "val.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print("Wrote:")
    print(f"  {train_out}  rows={len(train_df)} patients={len(train_patient_set)}")
    print(f"  {val_out}    rows={len(val_df)} patients={len(val_patient_set)}")

    # Also report image-path overlap (should be zero if patient overlap is zero, but cheap to show)
    train_paths = set(train_df["Path"].tolist())
    val_paths = set(val_df["Path"].tolist())
    path_overlap = len(train_paths & val_paths)
    print(f"Image path overlap: {path_overlap}")


if __name__ == "__main__":
    main()
