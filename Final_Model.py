import os
from pathlib import Path
import json
import math
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATASET = "ptbxl"
DATA_DIR = Path(f"Segments/{DATASET}")
OUTPUT_DIR = Path(f"StackedRecons_{DATASET}_Contr")
REPS_DIR = DATA_DIR / "reps" / "by_patient"
if not REPS_DIR.exists():
    REPS_DIR = Path("reps")
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "preds").mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "reports" / "evaluation_report.txt"

INPUT_LEADS = ["I", "II", "V2"]
TARGET_LEADS = ["V1", "V3", "V4", "V5", "V6"]
ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SEGMENT_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SHARD_PATTERN = "all_shard_*.npy"

REPS_BASE = [
    "clean_signal",
    "contrastive_raw_I_II_V2"
]

def load_all_shards(split_dir: Path):
    files = sorted(list(split_dir.glob(SHARD_PATTERN)))
    if len(files) == 0:
        raise FileNotFoundError(f"No shard files found in {split_dir}")
    shard_paths = []
    total = 0
    for p in files:
        try:
            arr = np.load(str(p), mmap_mode="r")
            cnt = int(arr.shape[0])
            del arr
        except Exception:
            cnt = 0
        if cnt > 0:
            shard_paths.append(str(p))
            total += cnt
    return shard_paths, total

def load_sample_from_shards(shard_paths, idx):
    acc = 0
    for p in shard_paths:
        arr = np.load(p, mmap_mode="r")
        l = arr.shape[0]
        if idx < acc + l:
            local = idx - acc
            s = np.array(arr[local], dtype=np.float32)
            del arr
            return s
        del arr
        acc += l
    if len(shard_paths) == 0:
        return np.zeros((SEGMENT_LENGTH, len(ALL_LEADS)), dtype=np.float32)
    arr0 = np.load(shard_paths[0], mmap_mode="r")
    out = np.zeros(arr0.shape[1:], dtype=np.float32)
    del arr0
    return out

def discover_rep_meta(rep_dir: Path, split: str):
    shards = sorted(list(rep_dir.glob(f"{split}_shard_*.npy")))
    if len(shards) == 0:
        return None
    total = 0
    sample_shape = None
    for s in shards:
        arr = np.load(str(s), mmap_mode="r")
        if sample_shape is None:
            sample_shape = list(arr.shape[1:])
        total += int(arr.shape[0])
        del arr
    return {"shards": [str(p) for p in shards], "num_samples": int(total), "sample_shape": sample_shape}

def load_rep_info(rep_name, split):
    rep_dir = Path(REPS_DIR) / rep_name
    meta_file = rep_dir / f"{split}_meta.json"
    if meta_file.exists():
        try:
            meta = json.load(open(meta_file, "r"))
            return {"shards": meta.get("shards", []), "num_samples": int(meta.get("num_samples", 0)), "sample_shape": meta.get("sample_shape")}
        except Exception:
            pass
    if rep_dir.exists():
        meta = discover_rep_meta(rep_dir, split)
        if meta is not None:
            try:
                with open(meta_file, "w") as fh:
                    json.dump(meta, fh)
            except Exception:
                pass
            return meta
    return None

def build_clean_signal_rep(split, shard_paths, out_dir: Path, input_leads=INPUT_LEADS, shard_size=4096):
    rep_dir = out_dir / "clean_signal"
    rep_dir.mkdir(parents=True, exist_ok=True)
    meta = {"shards": [], "num_samples": 0, "sample_shape": None}
    out_buf = []
    out_idx = 0
    shard_counts = []
    for p in shard_paths:
        arr = np.load(p, mmap_mode="r")
        shard_counts.append(int(arr.shape[0]))
        del arr
    for shard_i, cnt in enumerate(shard_counts):
        p = shard_paths[shard_i]
        arr = np.load(p, mmap_mode="r")
        for i in range(cnt):
            s = np.array(arr[i], dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            lead_cols = []
            for ld in input_leads:
                if ld in ALL_LEADS:
                    ch_idx = ALL_LEADS.index(ld)
                else:
                    ch_idx = 0
                lead_cols.append(s[:, ch_idx])
            sample = np.stack(lead_cols, axis=0).astype(np.float32)
            if meta["sample_shape"] is None:
                meta["sample_shape"] = list(sample.shape)
            out_buf.append(sample)
            meta["num_samples"] += 1
            if len(out_buf) >= shard_size:
                out_path = rep_dir / f"{split}_shard_{out_idx:04d}.npy"
                np.save(str(out_path), np.stack(out_buf, axis=0).astype(np.float32))
                meta["shards"].append(str(out_path))
                out_idx += 1
                out_buf = []
                gc.collect()
        del arr
    if len(out_buf) > 0:
        out_path = rep_dir / f"{split}_shard_{out_idx:04d}.npy"
        np.save(str(out_path), np.stack(out_buf, axis=0).astype(np.float32))
        meta["shards"].append(str(out_path))
        out_buf = []
    with open(rep_dir / f"{split}_meta.json", "w") as fh:
        json.dump(meta, fh)
    return meta

class MultiRepDataset(Dataset):
    def __init__(self, rep_info_map, y_shard_paths, reps, target_lead, split_name, N_max=None):
        self.rep_info_map = rep_info_map
        self.reps = reps
        self.target_lead = target_lead
        self.split_name = split_name
        N_all = min(int(rep_info_map[r]["num_samples"]) for r in reps)
        self.N = min(N_all, N_max) if N_max is not None else N_all
        self.y_shards = y_shard_paths
        self.rep_shards = {r: rep_info_map[r]["shards"] for r in reps}
    def __len__(self):
        return int(self.N)
    def __getitem__(self, idx):
        sample = {}
        for r in self.reps:
            shards = self.rep_shards[r]
            arr = load_sample_from_shards(shards, idx)
            if arr.ndim == 2:
                if arr.shape[0] == SEGMENT_LENGTH:
                    arr = arr.T
            sample[r] = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        seg = load_sample_from_shards(self.y_shards, idx)
        if seg.ndim == 2 and seg.shape[1] == len(ALL_LEADS):
            y = seg[:, ALL_LEADS.index(self.target_lead)]
        elif seg.ndim == 2 and seg.shape[0] == len(ALL_LEADS):
            y = seg[ALL_LEADS.index(self.target_lead), :]
        else:
            y = np.array(seg).ravel()[:SEGMENT_LENGTH]
        y = np.asarray(y, dtype=np.float32)
        return sample, y

class MultiRepIndicesDataset(Dataset):
    def __init__(self, rep_info_map, y_shard_paths, reps, target_lead, split_name, indices):
        self.rep_info_map = rep_info_map
        self.reps = reps
        self.target_lead = target_lead
        self.split_name = split_name
        self.indices = list(indices)
        self.y_shards = y_shard_paths
        self.rep_shards = {r: rep_info_map[r]["shards"] for r in reps}
    def __len__(self):
        return int(len(self.indices))
    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])
        sample = {}
        for r in self.reps:
            shards = self.rep_shards[r]
            arr = load_sample_from_shards(shards, global_idx)
            if arr.ndim == 2:
                if arr.shape[0] == SEGMENT_LENGTH:
                    arr = arr.T
            sample[r] = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        seg = load_sample_from_shards(self.y_shards, global_idx)
        if seg.ndim == 2 and seg.shape[1] == len(ALL_LEADS):
            y = seg[:, ALL_LEADS.index(self.target_lead)]
        elif seg.ndim == 2 and seg.shape[0] == len(ALL_LEADS):
            y = seg[ALL_LEADS.index(self.target_lead), :]
        else:
            y = np.array(seg).ravel()[:SEGMENT_LENGTH]
        y = np.asarray(y, dtype=np.float32)
        return sample, y

def normalize_arr(a):
    denom = a.std() + 1e-8
    if denom == 0:
        return a.astype(np.float32)
    return ((a - a.mean()) / denom).astype(np.float32)

def collate_batch(batch, reps, training=False):
    batch_inputs = {r: [] for r in reps}
    batch_y = []
    for sample, y in batch:
        for r in reps:
            a = sample.get(r)
            if a is None:
                a = np.zeros((len(INPUT_LEADS), SEGMENT_LENGTH), dtype=np.float32)
            if a.ndim == 2:
                if a.shape[0] == SEGMENT_LENGTH and a.shape[1] != SEGMENT_LENGTH:
                    a = a.T
                elif a.shape[1] == SEGMENT_LENGTH:
                    pass
                else:
                    raise RuntimeError(f"Unexpected rep shape {a.shape}")
            if r == "clean_signal":
                a = normalize_arr(a)
            batch_inputs[r].append(a)
        batch_y.append(y)
    coll = {}
    for r in reps:
        arrs = np.stack(batch_inputs[r], axis=0).astype(np.float32)
        coll[r] = torch.tensor(arrs)
    y_t = torch.tensor(np.stack(batch_y, axis=0).astype(np.float32))
    return coll, y_t

class SimplePerRepProj(nn.Module):
    def __init__(self, in_ch, per_branch_dim, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, per_branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(per_branch_dim),
            nn.GELU(),
            nn.Conv1d(per_branch_dim, per_branch_dim, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.seq_len = seq_len
    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] != self.seq_len:
            out = nn.functional.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out

class StackedLatentDecoder(nn.Module):
    def __init__(self, reps, rep_in_ch_map, per_branch_dim=128, seq_len=SEGMENT_LENGTH):
        super().__init__()
        self.reps = list(reps)
        self.seq_len = seq_len
        self.per_branch_dim = per_branch_dim
        self.proj = nn.ModuleDict()
        for r in self.reps:
            in_ch = int(rep_in_ch_map.get(r, 1))
            in_ch = max(1, in_ch)
            self.proj[r] = SimplePerRepProj(in_ch, per_branch_dim, seq_len)
        self.fusion = nn.Conv1d(len(self.reps)*per_branch_dim, per_branch_dim, kernel_size=3, padding=1, bias=False)
        self.decoder_type = "small"
        self.decoder = nn.Sequential(
            nn.Conv1d(per_branch_dim, max(32, per_branch_dim//2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(max(32, per_branch_dim//2), 1, kernel_size=1)
        )
    def forward(self, coll):
        B = None
        feats = []
        for r in self.reps:
            if r in coll:
                x = coll[r]
            else:
                in_ch = 1
                if hasattr(self, "rep_in_ch_map") and r in self.rep_in_ch_map:
                    in_ch = int(self.rep_in_ch_map[r])
                x = torch.zeros((B if B is not None else 1, in_ch, self.seq_len), device=next(self.parameters()).device)
            if B is None:
                B = x.shape[0]
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1,1,self.seq_len)
            if x.dim() == 3 and x.shape[-1] == self.seq_len and x.shape[1] == self.seq_len:
                x = x.permute(0,2,1)
            out = self.proj[r](x.float())
            feats.append(out)
        cat = torch.cat(feats, dim=1)
        fused = self.fusion(cat)
        out = self.decoder(fused)
        out = out.squeeze(1)
        return out

def pearson_mean(y_true, y_pred):
    vals = []
    for i in range(y_true.shape[0]):
        if np.std(y_true[i]) > 0:
            try:
                vals.append(pearsonr(y_true[i], y_pred[i])[0])
            except Exception:
                vals.append(0.0)
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

def compute_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "pearson": 0.0}
    rmse = math.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    try:
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
    except Exception:
        r2 = float("nan")
    pear = pearson_mean(y_true, y_pred)
    return {"rmse": rmse, "r2": r2, "pearson": pear}

def train_model(model, train_loader, val_loader, epochs, lr, device, model_tag, report_fh):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=False)
    patience = 12
    patience_ctr = 0
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for coll, y in tqdm(train_loader, desc=f"Train {model_tag} Epoch {epoch}/{epochs}", leave=False):
            for k in coll:
                coll[k] = coll[k].to(device)
            y = y.to(device)
            pred = model(coll)
            loss = 0.8*loss_fn(pred, y) + 0.2*torch.nn.functional.l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = pred.shape[0]
            epoch_loss += float(loss.item()) * bs
            n += bs
        train_loss = epoch_loss / max(1, n)
        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for coll, y in tqdm(val_loader, desc=f"Val {model_tag} Epoch {epoch}/{epochs}", leave=False):
                for k in coll:
                    coll[k] = coll[k].to(device)
                y = y.to(device)
                pred = model(coll)
                loss = loss_fn(pred, y)
                val_loss_sum += float(loss.item()) * pred.shape[0]
                val_preds.append(pred.cpu().numpy())
                val_trues.append(y.cpu().numpy())
        n_val = len(val_loader.dataset)
        val_loss = val_loss_sum / max(1, n_val)
        val_preds = np.vstack(val_preds) if len(val_preds) > 0 else np.zeros((0, SEGMENT_LENGTH))
        val_trues = np.vstack(val_trues) if len(val_trues) > 0 else np.zeros((0, SEGMENT_LENGTH))
        m = compute_metrics(val_trues, val_preds)
        line = f"[{model_tag}] Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_rmse={m['rmse']:.6f} val_r2={m['r2']:.6f} val_pear={m['pearson']:.6f}"
        print(line)
        if report_fh is not None:
            report_fh.write(line + "\n")
        try:
            scheduler.step(val_loss)
        except Exception:
            pass
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_model_numpy(model, loader, device=DEVICE):
    model.eval()
    preds = []
    with torch.no_grad():
        for coll, y in loader:
            for k in coll:
                coll[k] = coll[k].to(device)
            out = model(coll).cpu().numpy()
            preds.append(out)
    return np.vstack(preds) if len(preds) > 0 else np.zeros((0, SEGMENT_LENGTH))

def load_contrastive_nested(base, lead, split):
    base_dir = Path(REPS_DIR) / base
    if not base_dir.exists():
        return None
    nested_dir = base_dir / f"contrastive_h__{lead}"
    if not nested_dir.exists():
        return None
    meta_path = nested_dir / f"{split}_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.load(open(meta_path, "r"))
        return {"shards": meta.get("shards", []), "num_samples": int(meta.get("num_samples", 0)), "sample_shape": meta.get("sample_shape")}
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Train stacked decoder using fixed patient-wise folds")
    parser.add_argument("--folds-dir", type=str, default="folds_patientwise", help="Directory containing fixed folds (folds.npy or folds.json or fold*_train.npy)")
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    folds_dir = Path(args.folds_dir)
    out_dir = Path(args.out_dir)

    split_root = Path(DATA_DIR) / "by_patient"
    assert split_root.exists(), f"Expected split root at {split_root}"
    shard_paths_train, n_train = load_all_shards(split_root / "train")
    shard_paths_val, n_val = load_all_shards(split_root / "val")
    shard_paths_test, n_test = load_all_shards(split_root / "test")
    shard_paths_combined = shard_paths_train + shard_paths_val
    n_combined = n_train + n_val
    print(f"Found shards train={n_train} val={n_val} test={n_test}")
    print(f"[INFO] Combined train+val samples: {n_combined} (train {n_train} + val {n_val})")

    if not folds_dir.exists():
        raise RuntimeError(f"Folds directory {folds_dir} not found. Please run make_patient_folds.py first.")
    folds = None
    f_npy = folds_dir / "folds.npy"
    f_json = folds_dir / "folds.json"
    if f_npy.exists():
        folds = list(np.load(str(f_npy), allow_pickle=True))
        print(f"[INFO] Loaded folds from {f_npy}")
    elif f_json.exists():
        raw = json.load(open(str(f_json), "r"))
        folds = raw.get("folds", [])
        print(f"[INFO] Loaded folds from {f_json}")
    else:
        train_files = sorted(list(folds_dir.glob("fold*_train.npy")))
        val_files = sorted(list(folds_dir.glob("fold*_val.npy")))
        if len(train_files) != len(val_files) or len(train_files) == 0:
            raise RuntimeError(f"No usable folds found in {folds_dir}. Expected folds.npy or folds.json or fold*_train.npy + fold*_val.npy")
        folds = []
        for tfile, vfile in zip(train_files, val_files):
            train_idx = np.load(str(tfile))
            val_idx = np.load(str(vfile))
            folds.append({"fold": int(tfile.stem.replace("fold","").split("_")[0]), "train": train_idx.tolist(), "val": val_idx.tolist()})
        print(f"[INFO] Discovered {len(folds)} folds from individual files in {folds_dir}")

    K = len(folds)
    print(f"[INFO] Using K={K} fixed folds")

    rep_info_combined = {}
    rep_info_test = {}
    expanded_reps = []

    for base in REPS_BASE:
        if base == "clean_signal":
            expanded_reps.append("clean_signal")
            continue
        if base.startswith("contrastive"):
            for ld in INPUT_LEADS:
                expanded_reps.append(f"{base}__{ld}")
            continue
        for ld in INPUT_LEADS:
            key = f"{base}__{ld}"
            if load_rep_info(key, "train") is not None or load_rep_info(key, "val") is not None:
                expanded_reps.append(key)

    for r in expanded_reps:
        mt = load_rep_info(r, "train")
        if mt is None and r == "clean_signal":
            mt = build_clean_signal_rep("train", shard_paths_train, Path(REPS_DIR))
        if mt is None and r.startswith("contrastive"):
            base, ld = r.rsplit("__", 1)
            mt = load_contrastive_nested(base, ld, "train")
        mv = load_rep_info(r, "val")
        if mv is None and r == "clean_signal":
            mv = build_clean_signal_rep("val", shard_paths_val, Path(REPS_DIR))
        if mv is None and r.startswith("contrastive"):
            base, ld = r.rsplit("__", 1)
            mv = load_contrastive_nested(base, ld, "val")
        if mt is None and mv is None:
            raise RuntimeError(f"[COMBINED] Missing rep {r} in both train and val")
        combined_meta = {"shards": [], "num_samples": 0, "sample_shape": None}
        if mt is not None:
            combined_meta["shards"].extend(mt.get("shards", []))
            combined_meta["num_samples"] += int(mt.get("num_samples", 0))
            if combined_meta["sample_shape"] is None:
                combined_meta["sample_shape"] = mt.get("sample_shape")
        if mv is not None:
            combined_meta["shards"].extend(mv.get("shards", []))
            combined_meta["num_samples"] += int(mv.get("num_samples", 0))
            if combined_meta["sample_shape"] is None:
                combined_meta["sample_shape"] = mv.get("sample_shape")
        rep_info_combined[r] = combined_meta

        me = load_rep_info(r, "test")
        if me is None and r == "clean_signal":
            me = build_clean_signal_rep("test", shard_paths_test, Path(REPS_DIR))
        if me is None and r.startswith("contrastive"):
            base, ld = r.rsplit("__", 1)
            me = load_contrastive_nested(base, ld, "test")
        if me is None:
            raise RuntimeError(f"[TEST] Missing rep {r}")
        rep_info_test[r] = me

    Ns_combined = [rep_info_combined[r]["num_samples"] for r in expanded_reps]
    Ns_test = [rep_info_test[r]["num_samples"] for r in expanded_reps]
    N_combined = int(min(Ns_combined))
    N_test = int(min(Ns_test))
    print(f"Aligned counts -> combined(train+val)={N_combined}, test={N_test}")
    if N_combined == 0 or N_test == 0:
        raise RuntimeError("Not enough aligned samples across reps.")

    rep_in_ch_map = {}
    for r, m in rep_info_combined.items():
        shape = m.get("sample_shape")
        if shape is None:
            rep_in_ch_map[r] = 1
        else:
            rep_in_ch_map[r] = int(shape[0])

    n_common = None
    try:
        if (folds_dir / "folds.json").exists():
            meta = json.load(open(str(folds_dir / "folds.json"), "r"))
            n_common = int(meta.get("combined_train_val_num_samples", N_combined))
    except Exception:
        n_common = None
    if n_common is None:
        max_idx = 0
        for f in folds:
            max_idx = max(max_idx, max(f["train"]) if len(f["train"])>0 else 0, max(f["val"]) if len(f["val"])>0 else 0)
        n_common = int(max_idx) + 1

    print(f"[INFO] n_common from folds = {n_common}  (aligned N_combined = {N_combined})")
    if n_common > N_combined:
        raise RuntimeError(f"Folds reference indices up to {n_common-1} but aligned combined rep pool has only {N_combined} samples. Recreate folds with correct ordering/size.")

    metrics_across_folds = {lead: [] for lead in TARGET_LEADS}

    for fold_idx, fdict in enumerate(folds):
        fold_id = int(fdict.get("fold", fold_idx+1))
        train_idx = np.array(fdict["train"], dtype=int)
        val_idx = np.array(fdict["val"], dtype=int)
        print(f"\n=== Starting Fold {fold_id}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} ===")
        report_path = out_dir / "reports" / f"evaluation_report_fold{fold_id}.txt"

        if report_path.exists():
            print(f"[SKIP] Fold {fold_id} already completed (report exists). Skipping fold.")
            continue

        print(f"\n=== Starting Fold {fold_id}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} ===")
        with open(report_path, "w") as report:
            report.write(f"StackedLatent -> Decoder run - fixed patient-wise fold {fold_id}\nReps: {expanded_reps}\n")
            report.write(f"Fold sizes: train={len(train_idx)} N_val={len(val_idx)} test={N_test}\n")

            for lead in TARGET_LEADS:
                report.write(f"\n=== Lead {lead} (fold {fold_id}) ===\n")
                print(f"\n=== Training for lead {lead} (fold {fold_id}) ===")

                ds_train = MultiRepIndicesDataset(rep_info_combined, shard_paths_combined, expanded_reps, lead, "combined", train_idx.tolist())
                ds_val   = MultiRepIndicesDataset(rep_info_combined, shard_paths_combined, expanded_reps, lead, "combined", val_idx.tolist())
                ds_test  = MultiRepDataset(rep_info_test, shard_paths_test, expanded_reps, lead, "test", N_max=N_test)

                assert len(ds_train) == len(train_idx)
                assert len(ds_val) == len(val_idx)

                report.write(f"Using N_train={len(ds_train)} N_val={len(ds_val)} N_test={len(ds_test)}\n")
                train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, expanded_reps, training=True), num_workers=4, pin_memory=True)
                val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, expanded_reps, training=False), num_workers=4, pin_memory=True)
                test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, expanded_reps, training=False), num_workers=4, pin_memory=True)

                per_branch_dim_used = 64
                decoder_variant = "small"

                model = StackedLatentDecoder(expanded_reps, rep_in_ch_map, per_branch_dim=per_branch_dim_used, seq_len=SEGMENT_LENGTH)

                report.write(f"Model params: {sum(p.numel() for p in model.parameters())}\n")
                report.write(f"Decoder type: {decoder_variant}, per_branch_dim: {per_branch_dim_used}, num_reps: {len(expanded_reps)}\n")

                model_tag = f"Stacked_{lead}_fold{fold_id}"
                trained = train_model(model, train_loader, val_loader, args.epochs, LEARNING_RATE, DEVICE, model_tag, report)
                torch.save(trained.state_dict(), out_dir / "models" / f"{model_tag}.pt")

                preds_test = predict_model_numpy(trained, test_loader, device=DEVICE)
                y_list = []
                for i in range(len(ds_test)):
                    seg = load_sample_from_shards(shard_paths_test, i)
                    if seg.ndim == 2 and seg.shape[1] == len(ALL_LEADS):
                        y_list.append(seg[:, ALL_LEADS.index(lead)])
                    elif seg.ndim == 2 and seg.shape[0] == len(ALL_LEADS):
                        y_list.append(seg[ALL_LEADS.index(lead), :])
                    else:
                        y_list.append(np.array(seg).ravel()[:SEGMENT_LENGTH])
                y_test = np.stack(y_list, axis=0).astype(np.float32)

                m = compute_metrics(y_test, preds_test)
                line = f"[FINAL_TEST] Lead {lead} Fold {fold_id} RMSE={m['rmse']:.6f} R2={m['r2']:.6f} Pearson={m['pearson']:.6f}"
                print(line)
                report.write(line + "\n")
                metrics_across_folds[lead].append(m)

                out_pred_dir = out_dir / "preds" / f"{lead}_fold{fold_id}"
                out_pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_pred_dir / f"y_test_fold{fold_id}.npy", y_test)
                np.save(out_pred_dir / f"preds_test_fold{fold_id}.npy", preds_test)

                rmse_per_point = np.sqrt(np.mean((y_test - preds_test) ** 2, axis=0))
                plt.figure(figsize=(10,4))
                plt.plot(rmse_per_point)
                plt.title(f"RMSE per timepoint - Lead {lead} (fold {fold_id})")
                plt.xlabel("Time index")
                plt.ylabel("RMSE")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / "plots" / f"rmse_lead_{lead}_fold{fold_id}.png")
                plt.close()

                n_show = min(8, y_test.shape[0])
                fig, axes = plt.subplots(n_show, 1, figsize=(10, 2*n_show))
                for i in range(n_show):
                    axes[i].plot(y_test[i], label="true")
                    axes[i].plot(preds_test[i], linestyle='--', label="pred")
                    axes[i].set_xlim(0, SEGMENT_LENGTH-1)
                    axes[i].legend(fontsize=6)
                plt.tight_layout()
                plt.savefig(out_dir / "plots" / f"{model_tag}_overlay_first{n_show}.png", dpi=200)
                plt.close()

                for i in range(n_show):
                    true_sig = y_test[i]
                    pred_sig = preds_test[i]
                    t = np.arange(true_sig.shape[0])
                    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
                    ax1.plot(t, true_sig, linewidth=1.2, label='true')
                    ax1.plot(t, pred_sig, linewidth=1.0, linestyle='--', label='pred')
                    ax1.set_title(f"{model_tag} - sample {i} (full)")
                    ax1.grid(True)
                    ax1.legend()
                    center = true_sig.shape[0]//2
                    win = 80
                    s = max(0, center-win)
                    e = min(true_sig.shape[0], center+win)
                    ax2.plot(t[s:e], true_sig[s:e], linewidth=1.2, label='true')
                    ax2.plot(t[s:e], pred_sig[s:e], linewidth=1.0, linestyle='--', label='pred')
                    ax2.set_title(f"zoom [{s}:{e}]")
                    ax2.grid(True)
                    ax2.legend()
                    plt.tight_layout()
                    plt.savefig(out_pred_dir / f"{model_tag}_sample_{i:02d}.png", dpi=200)
                    plt.close()

                report.write(f"Saved predictions + plots for lead {lead} (fold {fold_id}) in {out_pred_dir}\n")
            report.write("\n")
        print(f"Finished fold {fold_id}. Report: {report_path}")

    summary_path = out_dir / "reports" / "evaluation_report_summary.txt"
    with open(summary_path, "w") as summ:
        summ.write(f"K-Fold Summary (predefined patient-wise folds K={len(folds)})\n")
        for lead in TARGET_LEADS:
            fold_metrics = metrics_across_folds.get(lead, [])
            if len(fold_metrics) == 0:
                summ.write(f"{lead}: no metrics collected\n")
                continue
            rmses = np.array([m['rmse'] for m in fold_metrics], dtype=float)
            r2s   = np.array([m['r2'] for m in fold_metrics], dtype=float)
            pears = np.array([m['pearson'] for m in fold_metrics], dtype=float)
            line = (f"{lead}: RMSE mean={rmses.mean():.6f} std={rmses.std(ddof=0):.6f} | "
                    f"R2 mean={np.nanmean(r2s):.6f} std={np.nanstd(r2s, ddof=0):.6f} | "
                    f"Pearson mean={pears.mean():.6f} std={pears.std(ddof=0):.6f}")
            print(line)
            summ.write(line + "\n")
    print("Fixed patient-wise folds CV finished. Summary report:", summary_path)
    print("Finished. Results in", out_dir)

if __name__ == "__main__":
    main()
