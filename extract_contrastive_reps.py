import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

SEGMENTS_ROOT = Path("./Segments")
LEAD_INDICES = [0, 1, 7]
INPUT_LEADS = ["I", "II", "V2"]
BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SHARD_SIZE = 4096
HIDDEN = 256
PROJ_DIM = 128
SEGMENT_LEN = 256

# Encoder
class ResNet1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
        else:
            self.down = nn.Identity()
    def forward(self, x):
        r = self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)

# Encoder
class ResNet1DEncoder(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, proj_dim=PROJ_DIM, depth=4):
        super().__init__()
        layers = []
        ch = 32
        layers.append(nn.Conv1d(in_ch, ch, 7, padding=3))
        layers.append(nn.BatchNorm1d(ch))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))
        for i in range(depth):
            layers.append(ResNet1DBlock(ch, ch*2 if i%2==0 else ch))
            if i%2==0:
                ch = ch*2
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.body = nn.Sequential(*layers)
        self.hidden = ch
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, proj_dim)
        )
    def forward(self, x):
        h = self.body(x).squeeze(-1)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z
    def encode_h(self, x):
        h = self.body(x).squeeze(-1)
        return h

# Checkpoint
def robust_load_checkpoint(path):
    ck = torch.load(str(path), map_location="cpu")
    if isinstance(ck, dict):
        if "model_state" in ck:
            return ck["model_state"]
        if "model_state_dict" in ck:
            return ck["model_state_dict"]
        if "state_dict" in ck:
            return ck["state_dict"]
        return ck
    return ck

# IO
def list_shards_for_split(dataset, split):
    folder = SEGMENTS_ROOT / dataset / "by_patient" / split
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("all_shard_*.npy") if p.is_file()])

# IO
def list_shards_for_all(dataset):
    folder = SEGMENTS_ROOT / dataset / "all"
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("all_shard_*.npy") if p.is_file()])

# IO
def meta_for_shard(shard_path):
    p = Path(shard_path)
    meta_path = p.with_name(p.stem + "_meta.json")
    if not meta_path.exists():
        candidates = list(p.parent.glob(p.stem + "*_meta.json"))
        if len(candidates) > 0:
            return candidates[0]
        return None
    return meta_path

# Utils
def total_samples_in_shards_using_meta(shard_list):
    total = 0
    for p in shard_list:
        m = meta_for_shard(p)
        if m is not None:
            try:
                with open(m, 'r', encoding='utf-8') as fh:
                    ml = json.load(fh)
                total += len(ml)
                continue
            except Exception:
                pass
        try:
            total += int(np.load(str(p), mmap_mode='r').shape[0])
        except Exception:
            continue
    return total

# Encoder save
def encode_split_and_save(split_name, shard_files, encoder, out_base,
                          batch_size=BATCH, device="cpu", shard_size=DEFAULT_SHARD_SIZE,
                          tile=True, save_global=False, reps_dest_path=None):
    device = torch.device(device)
    encoder.to(device)
    encoder.eval()
    reps_dest = Path(reps_dest_path) if reps_dest_path is not None else Path("reps/contrastive_raw_I_II_V2")
    reps_dest.mkdir(parents=True, exist_ok=True)
    per_lead_dirs = {ld: (reps_dest / f"contrastive_h__{ld}") for ld in INPUT_LEADS}
    for d in per_lead_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    if save_global:
        gv_dir = Path(out_base) / "h_vectors"
        gv_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(out_base) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved_buffers = {ld: [] for ld in INPUT_LEADS}
    saved_shards = {ld: [] for ld in INPUT_LEADS}
    saved_id_bufs = {ld: [] for ld in INPUT_LEADS}
    shard_idx_map = {ld: 0 for ld in INPUT_LEADS}
    gv_buf = []
    gv_shards = []
    gv_idx = 0
    labels_all = []
    sample_ids_all = []
    total_samples = total_samples_in_shards_using_meta(shard_files)
    pbar = tqdm(total=total_samples, desc=f"Encode {split_name}")
    processed_count = 0
    for shard_path in shard_files:
        try:
            arr = np.load(shard_path, mmap_mode="r")
        except Exception as e:
            print(f"[WARN] cannot load {shard_path}: {e}")
            continue
        meta_p = meta_for_shard(shard_path)
        metas = None
        if meta_p is not None:
            try:
                with open(meta_p, 'r', encoding='utf-8') as fh:
                    metas = json.load(fh)
            except Exception:
                metas = None
        N = int(arr.shape[0])
        if metas is not None:
            n_meta = len(metas)
            if n_meta < N:
                print(f"[WARN] meta length {n_meta} < shard samples {N} for {shard_path}. Only encoding first {n_meta} samples.")
            elif n_meta > N:
                print(f"[WARN] meta length {n_meta} > shard samples {N} for {shard_path}. Using available {N} samples.")
            n_local = min(N, n_meta)
        else:
            print(f"[WARN] No meta found for shard {shard_path}. Falling back to shard length {N}. Consider regenerating meta.")
            n_local = N
        start_idx = 0
        while start_idx < n_local:
            end_idx = min(start_idx + batch_size, n_local)
            batch = arr[start_idx:end_idx]
            if batch.ndim == 3 and batch.shape[2] >= max(LEAD_INDICES)+1:
                sel = batch[:, :, LEAD_INDICES].astype(np.float32)
            elif batch.ndim == 3 and batch.shape[1] >= max(LEAD_INDICES)+1:
                sel = batch[:, LEAD_INDICES, :].astype(np.float32)
                sel = np.transpose(sel, (0, 2, 1))
            else:
                Bf = batch.shape[0]
                sel = np.zeros((Bf, SEGMENT_LEN, len(LEAD_INDICES)), dtype=np.float32)
            x = torch.from_numpy(sel).permute(0, 2, 1).to(device=device)
            with torch.no_grad():
                h = encoder.encode_h(x)
            h_np = h.cpu().numpy().astype(np.float32)
            Bc = h_np.shape[0]
            for bi in range(Bc):
                global_local_idx = start_idx + bi
                lbl = None
                rec_id = None
                if metas is not None and global_local_idx < len(metas):
                    meta_entry = metas[global_local_idx]
                    lbl = meta_entry.get("super_class") or meta_entry.get("primary_scp") or None
                    rec_id = meta_entry.get("rec_id") or meta_entry.get("record_id") or None
                z = h_np[bi]
                z_tile = np.tile(z[:, None], (1, sel.shape[1])).astype(np.float32) if tile else None
                sample_id = {"shard": str(Path(shard_path).name), "local_idx": int(global_local_idx)}
                if rec_id is not None:
                    sample_id["rec_id"] = str(rec_id)
                for ld in INPUT_LEADS:
                    if tile:
                        saved_buffers[ld].append(z_tile)
                        saved_id_bufs[ld].append(sample_id)
                        if len(saved_buffers[ld]) >= shard_size:
                            fname = f"{split_name}_shard_{shard_idx_map[ld]:04d}.npy"
                            outp = per_lead_dirs[ld] / fname
                            np.save(outp, np.stack(saved_buffers[ld], axis=0))
                            saved_shards[ld].append(str(outp))
                            id_fname = f"{split_name}_shard_{shard_idx_map[ld]:04d}_ids.json"
                            with open(per_lead_dirs[ld] / id_fname, 'w', encoding='utf-8') as fh:
                                json.dump(saved_id_bufs[ld][-shard_size:], fh, indent=2)
                            shard_idx_map[ld] += 1
                            saved_buffers[ld] = []
                            saved_id_bufs[ld] = []
                if save_global:
                    gv_buf.append(z)
                    labels_all.append(str(lbl) if lbl is not None else "None")
                    sample_ids_all.append(sample_id)
                    if len(gv_buf) >= shard_size:
                        fname = f"{split_name}_h_shard_{gv_idx:04d}.npy"
                        outp = gv_dir / fname
                        np.save(outp, np.stack(gv_buf, axis=0))
                        gv_shards.append(str(outp))
                        gv_idx += 1
                        gv_buf = []
                        with open(Path(out_base) / f"{split_name}_h_labels_{gv_idx-1:04d}.json", "w") as fh:
                            json.dump(labels_all[-shard_size:], fh)
                processed_count += 1
                pbar.update(1)
            start_idx = end_idx
        del arr
    for ld in INPUT_LEADS:
        if len(saved_buffers[ld]) > 0:
            fname = f"{split_name}_shard_{shard_idx_map[ld]:04d}.npy"
            outp = per_lead_dirs[ld] / fname
            np.save(outp, np.stack(saved_buffers[ld], axis=0))
            saved_shards[ld].append(str(outp))
            id_fname = f"{split_name}_shard_{shard_idx_map[ld]:04d}_ids.json"
            with open(per_lead_dirs[ld] / id_fname, 'w', encoding='utf-8') as fh:
                json.dump(saved_id_bufs[ld], fh, indent=2)
            saved_buffers[ld] = []
            saved_id_bufs[ld] = []
    if save_global and len(gv_buf) > 0:
        fname = f"{split_name}_h_shard_{gv_idx:04d}.npy"
        outp = gv_dir / fname
        np.save(outp, np.stack(gv_buf, axis=0))
        gv_shards.append(str(outp))
        with open(Path(out_base) / f"{split_name}_h_labels_final.json", "w") as fh:
            json.dump(labels_all, fh, indent=2)
        with open(Path(out_base) / f"{split_name}_h_sample_ids.json", "w") as fh:
            json.dump(sample_ids_all, fh, indent=2)
    pbar.close()
    metas_dict = {}
    for ld in INPUT_LEADS:
        if len(saved_shards[ld]) == 0:
            meta = {"shards": [], "dtype": "float32", "num_samples": 0, "sample_shape": [0]}
        else:
            num = sum([int(np.load(s, mmap_mode="r").shape[0]) for s in saved_shards[ld]])
            s0 = np.load(saved_shards[ld][0], mmap_mode="r")
            sample_shape = list(s0.shape[1:])
            meta = {"shards": saved_shards[ld], "dtype": "float32", "num_samples": int(num), "sample_shape": sample_shape}
        with open(Path(per_lead_dirs[ld]) / f"{split_name}_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        metas_dict[ld] = meta
    if save_global:
        if len(gv_shards) == 0:
            gmeta = {"shards": [], "dtype": "float32", "num_samples": 0, "sample_shape": [0]}
        else:
            num = sum([int(np.load(s, mmap_mode="r").shape[0]) for s in gv_shards])
            s0 = np.load(gv_shards[0], mmap_mode="r")
            sample_shape = list(s0.shape[1:])
            gmeta = {"shards": gv_shards, "dtype": "float32", "num_samples": int(num), "sample_shape": sample_shape}
        with open(Path(out_base) / f"{split_name}_h_vectors_meta.json", "w") as fh:
            json.dump(gmeta, fh, indent=2)
    with open(Path(reps_dest) / f"{split_name}_sample_ids.json", "w") as fh:
        json.dump(sample_ids_all, fh, indent=2)
    return metas_dict

# Diagnostics
def compute_diagnostics(out_base, split_name="train", limit_samples=20000, reports_dir=None):
    gv_dir = Path(out_base) / "h_vectors"
    shards = sorted([p for p in gv_dir.glob(f"{split_name}_h_shard_*.npy")])
    if len(shards) == 0:
        print("[WARN] No global h shards found for diagnostics.")
        return {}
    vecs = []
    for s in shards:
        vecs.append(np.load(s, mmap_mode="r"))
    vecs = np.vstack(vecs)
    with open(Path(out_base) / f"{split_name}_h_labels_final.json", "r") as fh:
        labels = json.load(fh)
    n = vecs.shape[0]
    if n > limit_samples:
        rng = np.random.RandomState(0)
        idxs = rng.choice(np.arange(n), size=limit_samples, replace=False)
        vecs_s = vecs[idxs]
        labels_s = [labels[i] for i in idxs]
    else:
        vecs_s = vecs
        labels_s = labels
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    unique = sorted(set(labels_s))
    lbl_to_idx = {l:i for i,l in enumerate(unique)}
    y = np.array([lbl_to_idx[l] for l in labels_s])
    perm = np.random.permutation(len(y))
    tr = perm[: int(0.8*len(y))]
    te = perm[int(0.8*len(y)):]
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf.fit(vecs_s[tr], y[tr])
    preds = clf.predict(vecs_s[te])
    acc = accuracy_score(y[te], preds)
    out = {"linear_probe_acc": float(acc)}
    repd = Path(reports_dir) if reports_dir is not None else Path(out_base) / "reports"
    repd.mkdir(parents=True, exist_ok=True)
    with open(repd / f"diagnostics_{split_name}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ptbxl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="out/contrastive_reps")
    parser.add_argument("--reps-dest", type=str,
                        default="reps/contrastive_raw_I_II_V2")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--tile", action="store_true", default=True)
    parser.add_argument("--no-tile", action="store_false", dest="tile")
    parser.add_argument("--save-global", action="store_true")
    args = parser.parse_args()
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    encoder = ResNet1DEncoder(in_ch=len(LEAD_INDICES), hidden=HIDDEN, proj_dim=PROJ_DIM, depth=4)
    state = robust_load_checkpoint(Path(args.checkpoint))
    try:
        encoder.load_state_dict(state, strict=True)
    except Exception:
        fixed = {}
        for k,v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            fixed[nk] = v
        encoder.load_state_dict(fixed, strict=False)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    reps_dest = Path(args.reps_dest)
    reps_dest.mkdir(parents=True, exist_ok=True)
    if args.dataset == "ptb":
        splits = ["all"]
    else:
        splits = ["train", "val", "test"]
    for sp in splits:
        if sp == "all":
            shards = list_shards_for_all(args.dataset)
        else:
            shards = list_shards_for_split(args.dataset, sp)
        if len(shards) == 0:
            print(f"[INFO] no shards for split {sp}, skipping.")
            continue
        total_n = total_samples_in_shards_using_meta(shards)
        print(f"[INFO] Encoding split {sp}: expected samples (from meta if present)={total_n}")
        encode_split_and_save(sp, shards, encoder, out_base,
                              batch_size=args.batch, device=device, shard_size=args.shard_size,
                              tile=args.tile, save_global=args.save_global,
                              reps_dest_path=str(reps_dest))
        if args.save_global and sp != "all":
            print(f"[INFO] computing diagnostics for split {sp} ...")
            d = compute_diagnostics(out_base, split_name=sp, reports_dir=str(out_base / "reports"))
            print("Diagnostics:", d)

if __name__ == "__main__":
    main()
