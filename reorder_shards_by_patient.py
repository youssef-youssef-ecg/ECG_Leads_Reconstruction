import argparse
import json
import glob
import random
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

# Load metas
def load_meta_list(all_meta_files):
    metas = []
    for mf in sorted(all_meta_files):
        with open(mf, 'r', encoding='utf-8') as f:
            m = json.load(f)
        if isinstance(m, dict):
            metas.append(_normalize_single_meta(m))
        elif isinstance(m, list):
            for item in m:
                metas.append(_normalize_single_meta(item))
    return metas

# Normalize
def _normalize_single_meta(m):
    if not isinstance(m, dict):
        return {}
    out = dict(m)
    if "rec_id" not in out or out.get("rec_id") is None:
        sp = out.get("source_path", None)
        if sp:
            out["rec_id"] = Path(sp).stem
        else:
            out["rec_id"] = out.get("record_id", out.get("id", "unknown"))
    if "patient_id" not in out or out.get("patient_id") in (None, ""):
        for key in ("patient", "patient_id", "patientId", "pid"):
            if key in out and out.get(key):
                out["patient_id"] = out.get(key)
                break
        else:
            out["patient_id"] = out["rec_id"]
    if "fs" not in out:
        if "sampling_rate" in out:
            out["fs"] = out["sampling_rate"]
        else:
            out["fs"] = out.get("sampling_freq", None)
    out.setdefault("scp_codes_raw", out.get("scp_codes_raw", None))
    out.setdefault("scp_codes", out.get("scp_codes", {}) or {})
    out.setdefault("primary_scp", out.get("primary_scp", None))
    out.setdefault("super_class", out.get("super_class", None))
    out.setdefault("diagnostic_subclass", out.get("diagnostic_subclass", None))
    out.setdefault("diagnosis_snomed_codes", out.get("diagnosis_snomed_codes", []) or [])
    out.setdefault("diagnosis_acronyms", out.get("diagnosis_acronyms", []) or [])
    out.setdefault("diagnosis_fullnames", out.get("diagnosis_fullnames", []) or [])
    return out

# Patients map
def build_patient_to_indices(metas):
    patient_to_idxs = OrderedDict()
    for i, m in enumerate(metas):
        pid = m.get("patient_id", None)
        if pid is None or pid == "":
            pid = "__NO_PID__"
        if pid not in patient_to_idxs:
            patient_to_idxs[pid] = []
        patient_to_idxs[pid].append(i)
    return patient_to_idxs

# Sample loader
def load_sample_from_shards(shard_paths, idx):
    acc = 0
    for p in shard_paths:
        arr = np.load(p, mmap_mode='r')
        l = int(arr.shape[0])
        if idx < acc + l:
            local = idx - acc
            return np.array(arr[local], dtype=np.float32)
        acc += l
    arr0 = np.load(shard_paths[0], mmap_mode="r")
    return np.zeros(arr0.shape[1:], dtype=np.float32)

# Greedy assign
def greedy_assign_patients(patient_to_idxs, target_counts, seed=42):
    rnd = random.Random(seed)
    patients = list(patient_to_idxs.keys())
    pid_counts = {pid: len(patient_to_idxs[pid]) for pid in patients}
    patients_sorted = sorted(patients, key=lambda p: (-pid_counts[p], str(p)))
    splits = ["train", "val", "test"]
    assigned = {s: [] for s in splits}
    split_current = {s: 0 for s in splits}
    targets = target_counts.copy()
    for pid in patients_sorted:
        best_split = None
        best_metric = None
        cnt = pid_counts[pid]
        for s in splits:
            tgt = targets.get(s, 0)
            if tgt <= 0:
                metric = split_current[s] + cnt
            else:
                metric = (split_current[s] + cnt) / tgt
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_split = s
        assigned[best_split].append(pid)
        split_current[best_split] += cnt
    return assigned, split_current

# Build ordered
def build_ordered_indices_from_patients(patient_list, patient_to_idxs):
    ordered = []
    for pid in patient_list:
        ordered.extend(patient_to_idxs[pid])
    return ordered

# Write shards
def write_shards(out_split_dir, ordered_indices, metas, shard_size, read_shard_paths, split_name):
    out_split_dir.mkdir(parents=True, exist_ok=True)
    meta_flat = metas
    shard_idx = 0
    buf_samples = []
    buf_meta = []
    written_shards = []
    pbar = tqdm(ordered_indices, desc=f"Writing {split_name}", leave=True)
    for gi in pbar:
        sample = load_sample_from_shards(read_shard_paths, gi)
        buf_samples.append(sample.astype(np.float32))
        meta_entry = meta_flat[gi].copy()
        for k, v in list(meta_entry.items()):
            if isinstance(v, (np.float32, np.float64)):
                meta_entry[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                meta_entry[k] = int(v)
        buf_meta.append(meta_entry)
        if len(buf_samples) >= shard_size:
            shard_name = f"all_shard_{shard_idx:04d}.npy"
            meta_name = f"all_shard_{shard_idx:04d}_meta.json"
            np.save(out_split_dir / shard_name, np.stack(buf_samples, axis=0))
            with open(out_split_dir / meta_name, "w", encoding='utf-8') as fh:
                json.dump(buf_meta, fh, ensure_ascii=False)
            written_shards.append(str(out_split_dir / shard_name))
            shard_idx += 1
            buf_samples = []
            buf_meta = []
    if len(buf_samples) > 0:
        shard_name = f"all_shard_{shard_idx:04d}.npy"
        meta_name = f"all_shard_{shard_idx:04d}_meta.json"
        np.save(out_split_dir / shard_name, np.stack(buf_samples, axis=0))
        with open(out_split_dir / meta_name, "w", encoding='utf-8') as fh:
            json.dump(buf_meta, fh, ensure_ascii=False)
        written_shards.append(str(out_split_dir / shard_name))
    return written_shards

# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--shard_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    out_base = Path(args.out)
    assert src.exists(), f"Source directory not found: {src}"
    out_base.mkdir(parents=True, exist_ok=True)

    shard_npy = sorted(glob.glob(str(src / "all_shard_*.npy")))
    shard_meta = sorted(glob.glob(str(src / "all_shard_*_meta.json")))
    if len(shard_npy) == 0 or len(shard_meta) == 0:
        raise SystemExit(f"No shard npy or meta files found in {src}")

    metas = load_meta_list(shard_meta)
    total = len(metas)
    patient_to_idxs = build_patient_to_indices(metas)

    n_train_target = int(round(0.70 * total))
    n_val_target = int(round(0.15 * total))
    n_test_target = total - n_train_target - n_val_target
    targets = {"train": n_train_target, "val": n_val_target, "test": n_test_target}

    assigned, split_counts = greedy_assign_patients(patient_to_idxs, targets, seed=args.seed)

    train_ordered = build_ordered_indices_from_patients(assigned["train"], patient_to_idxs)
    val_ordered = build_ordered_indices_from_patients(assigned["val"], patient_to_idxs)
    test_ordered = build_ordered_indices_from_patients(assigned["test"], patient_to_idxs)

    tot_now = len(train_ordered) + len(val_ordered) + len(test_ordered)
    assert tot_now == total, f"Split indices sum mismatch: {tot_now} vs {total}"

    for split_name, ordered in [("train", train_ordered), ("val", val_ordered), ("test", test_ordered)]:
        out_split_dir = out_base / split_name
        written_shards = write_shards(out_split_dir, ordered, metas, args.shard_size, shard_npy, split_name)
        if len(written_shards) == 0:
            meta = {"shards": [], "dtype": "float32", "num_samples": 0, "sample_shape": [0]}
        else:
            num = sum([int(np.load(s, mmap_mode="r").shape[0]) for s in written_shards])
            s0 = np.load(written_shards[0], mmap_mode="r")
            sample_shape = list(s0.shape[1:])
            meta = {"shards": written_shards, "dtype": "float32", "num_samples": int(num), "sample_shape": sample_shape}
        meta_path = out_base / f"{split_name}_meta.json"
        with open(meta_path, "w", encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    final_counts = {
        "train": int(len(train_ordered)),
        "val": int(len(val_ordered)),
        "test": int(len(test_ordered))
    }
    with open(out_base / "split_counts.json", "w", encoding='utf-8') as fh:
        json.dump(final_counts, fh, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
