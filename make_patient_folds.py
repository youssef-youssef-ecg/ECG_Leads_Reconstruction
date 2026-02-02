import argparse
from pathlib import Path
import json
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import GroupKFold

# Config
def load_meta_list(meta_files):
    metas = []
    for mf in sorted(meta_files):
        with open(mf, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                metas.append(item)
        elif isinstance(obj, dict):
            metas.append(obj)
        else:
            raise RuntimeError(f"Unexpected meta content in {mf}")
    return metas

# Collect shards
def collect_shards_and_meta(split_dir):
    p = Path(split_dir)
    shard_npy = sorted([str(x) for x in p.glob("all_shard_*.npy")])
    shard_meta = sorted([str(x) for x in p.glob("all_shard_*_meta.json")])
    return shard_npy, shard_meta

# Build mapping
def build_patient_to_indices(metas):
    patient_to_idxs = OrderedDict()
    for i, m in enumerate(metas):
        pid = m.get("patient_id", None) or m.get("patient", None) or m.get("pid", None) or m.get("patientId", None)
        if pid is None or pid == "":
            pid = m.get("rec_id", m.get("record_id", f"unknown_{i}"))
        if pid not in patient_to_idxs:
            patient_to_idxs[pid] = []
        patient_to_idxs[pid].append(i)
    return patient_to_idxs

# Write output
def write_folds_output(out_dir, folds, meta_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in folds:
        fid = f["fold"]
        np.save(out_dir / f"fold{fid}_train.npy", np.array(f["train"], dtype=np.int32))
        np.save(out_dir / f"fold{fid}_val.npy",   np.array(f["val"], dtype=np.int32))
    np.save(out_dir / "folds.npy", folds, allow_pickle=True)
    with open(out_dir / "folds.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta_info, "folds": folds}, fh, indent=2, ensure_ascii=False)

# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ptbxl")
    p.add_argument("--data_root", type=str, default="Segments", help="Root containing Segments/<dataset>/by_patient/<split>")
    p.add_argument("--k", type=int, default=5, help="Number of folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="folds_patientwise")
    args = p.parse_args()

    dataset = args.dataset
    k = int(args.k)
    seed = int(args.seed)
    out_dir = Path(args.out_dir)

    base_by_patient = Path(args.data_root) / dataset / "by_patient"
    train_dir = base_by_patient / "train"
    val_dir   = base_by_patient / "val"
    test_dir  = base_by_patient / "test"

    if not train_dir.exists():
        raise SystemExit(f"Train dir not found: {train_dir}")
    if not val_dir.exists():
        raise SystemExit(f"Val dir not found: {val_dir}")
    if not test_dir.exists():
        raise SystemExit(f"Test dir not found: {test_dir}")

    _, train_meta_files = collect_shards_and_meta(train_dir)
    _, val_meta_files   = collect_shards_and_meta(val_dir)

    if len(train_meta_files) == 0 and len(val_meta_files) == 0:
        raise SystemExit("No meta json files found under train/val. Aborting.")

    metas_train = load_meta_list(train_meta_files) if train_meta_files else []
    metas_val   = load_meta_list(val_meta_files)   if val_meta_files else []

    metas_combined = metas_train + metas_val
    total_samples = len(metas_combined)

    patient_to_idxs = build_patient_to_indices(metas_combined)
    n_patients = len(patient_to_idxs)

    indices = np.arange(total_samples)
    groups = []
    for i, m in enumerate(metas_combined):
        pid = m.get("patient_id", None) or m.get("patient", None) or m.get("pid", None) or m.get("patientId", None)
        if pid is None or pid == "":
            pid = m.get("rec_id", m.get("record_id", f"unknown_{i}"))
        groups.append(pid)
    groups = np.array(groups, dtype=object)

    if len(np.unique(groups)) < k:
        raise SystemExit(f"Number of unique patients ({len(np.unique(groups))}) < k ({k}). Reduce k or reassign splits.")

    gkf = GroupKFold(n_splits=k)
    folds = []
    fid = 0
    for train_idx, val_idx in gkf.split(indices, groups=groups):
        fid += 1
        folds.append({
            "fold": fid,
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx))
        })

    meta_info = {
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "combined_train_val_num_samples": total_samples,
        "train_samples": len(metas_train),
        "val_samples": len(metas_val),
        "n_patients": n_patients,
        "patient_to_sample_counts": {pid: len(idxs) for pid, idxs in patient_to_idxs.items()}
    }

    write_folds_output(out_dir, folds, meta_info)

    with open(out_dir / "combined_index_to_patient.json", "w", encoding="utf-8") as fh:
        json.dump({"index_to_patient": [str(p) for p in groups.tolist()]}, fh, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
