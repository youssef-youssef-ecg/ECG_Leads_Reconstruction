import argparse
import json
import os
import random
import time
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


DEFAULT_DATASET = "ptbxl"
SEGMENTS_ROOT = Path("./Segments")
LEAD_INDICES = [0, 1, 7]
SEGMENT_LEN = 256

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_r_peaks(sig_1d, fs=100):
    try:
        peaks, props = find_peaks(sig_1d, distance=int(0.2*fs), prominence=max(0.1, np.std(sig_1d)*0.4))
        return peaks, props
    except Exception:
        return np.array([]), {}


def find_shards_and_meta_for_split(dataset, split):
    base = SEGMENTS_ROOT / dataset / "by_patient" / split
    if not base.exists():
        raise FileNotFoundError(f"Split folder not found: {base}")
    npys = sorted([p for p in base.glob("*.npy") if "_shard" in p.name])
    metas = sorted([p for p in base.glob("*_meta.json") if p.name.endswith("_meta.json")])
    if len(npys) == 0:
        raise RuntimeError(f"No npy shards found in {base}")
    if len(metas) == 0:
        raise RuntimeError(f"No meta json files found in {base}")
    meta_map = {p.name.replace("_meta.json", ""): p for p in metas}
    mapping = []
    for n in npys:
        key = n.stem
        m = meta_map.get(key, None)
        if m is None:
            candidates = [p for p in metas if p.name.startswith(key)]
            m = candidates[0] if candidates else None
        mapping.append((n, m))
    return mapping

def build_index_from_mapping(mapping, skip_null=True):
    index_map = []
    class_counter = Counter()
    for npy_path, meta_path in mapping:
        with open(meta_path, "r", encoding="utf-8") as fh:
            mlist = json.load(fh)
        arr = np.load(npy_path, mmap_mode="r")
        n_local = arr.shape[0]
        n_local = min(n_local, len(mlist))
        for i in range(n_local):
            rec_meta = mlist[i]
            label = (
                rec_meta.get("super_class")
                or rec_meta.get("primary_scp")
                or (rec_meta.get("diagnosis_acronyms")[0] if rec_meta.get("diagnosis_acronyms") else None)
            )

            label_val = str(label) if label is not None else None
            if label_val is not None:
                class_counter[label_val] += 1
            entry = {"shard": str(npy_path), "local_idx": int(i), "label": label_val, "meta": rec_meta}
            index_map.append(entry)
    if skip_null:
        index_map = [e for e in index_map if e["label"] is not None]
    return index_map, class_counter


def augment_segment_basic(x, noise_std=0.01, scaling_std=0.08, shift_max=8, channel_drop_prob=0.1):
    x = x.copy()
    if noise_std and noise_std > 0:
        x = x + np.random.normal(0, noise_std, size=x.shape).astype(x.dtype)
    if scaling_std and scaling_std > 0:
        scales = np.random.normal(1.0, scaling_std, size=(x.shape[0], 1)).astype(x.dtype)
        x = x * scales
    if shift_max and shift_max > 0:
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift != 0:
            x = np.roll(x, shift, axis=1)
    if channel_drop_prob and channel_drop_prob > 0:
        mask = (np.random.rand(x.shape[0]) < channel_drop_prob)
        for ci, m in enumerate(mask):
            if m:
                x[ci, :] = 0.0
    return x

def augment_segment_beat_aligned(x, fs=100, noise_std=0.01, scaling_std=0.05, shift_max=4, channel_drop_prob=0.08):
    ch = 1 if x.shape[0] > 1 else 0
    peaks, _ = detect_r_peaks(x[ch, :], fs=fs)
    if len(peaks) >= 1:
        p = int(np.random.choice(peaks))
        center = SEGMENT_LEN // 2
        shift = center - p
        x = np.roll(x, shift, axis=1)
    return augment_segment_basic(x, noise_std=noise_std, scaling_std=scaling_std, shift_max=shift_max, channel_drop_prob=channel_drop_prob)

class SplitSegmentsDataset(Dataset):
    def __init__(self, index_map, lead_indices=[0,1,7], augment='basic'):
        self.index_map = index_map
        self.lead_indices = lead_indices
        self.shard_cache = {}
        self.augment = augment
    def __len__(self):
        return len(self.index_map)
    def _load_shard(self, path):
        if path not in self.shard_cache:
            self.shard_cache[path] = np.load(path, mmap_mode="r")
        return self.shard_cache[path]
    def __getitem__(self, idx):
        ent = self.index_map[idx]
        shard_path = ent["shard"]
        local = ent["local_idx"]
        arr = self._load_shard(shard_path)
        seg = np.asarray(arr[local], dtype=np.float32)   # (T, 12)
        seg = seg[:, self.lead_indices].T               # (C, T)
        if self.augment == 'beat':
            v1 = augment_segment_beat_aligned(seg)
            v2 = augment_segment_basic(seg)
        else:
            v1 = augment_segment_basic(seg)
            v2 = augment_segment_basic(seg)
        return v1, v2, idx

# ---------------- Balanced Batch Sampler ----------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, classes_per_batch=16, samples_per_class=4):
        self.labels = list(labels)
        self.labels_arr = np.array(self.labels)
        self.label_to_indices = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.label_to_indices[lab].append(idx)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.unique_labels = list(self.label_to_indices.keys())
        self.num_samples = len(self.labels)
        self.batches_per_epoch = max(1, int(self.num_samples / (classes_per_batch * samples_per_class)))
    def __len__(self):
        return self.batches_per_epoch
    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_classes = random.sample(self.unique_labels, min(self.classes_per_batch, len(self.unique_labels)))
            batch_indices = []
            for c in chosen_classes:
                idxs = self.label_to_indices[c]
                if len(idxs) >= self.samples_per_class:
                    chosen = random.sample(idxs, self.samples_per_class)
                else:
                    chosen = list(np.random.choice(idxs, size=self.samples_per_class, replace=True))
                batch_indices.extend(chosen)
            random.shuffle(batch_indices)
            yield batch_indices

# ---------------- Model building blocks ----------------
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

class ResNet1DEncoder(nn.Module):
    def __init__(self, in_ch, hidden=256, proj_dim=128, depth=4):
        super().__init__()
        layers = []
        ch = 32
        layers.append(nn.Conv1d(in_ch, ch, 7, padding=3))
        layers.append(nn.BatchNorm1d(ch)); layers.append(nn.ReLU()); layers.append(nn.MaxPool1d(2))
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

# ---------------- Loss (SupCon) ----------------
class SupConLossTorch(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, group_ids):
        # group_ids: list or tensor where same value => positive pair
        device = features.device
        if isinstance(group_ids, torch.Tensor):
            gids = group_ids.cpu().numpy().tolist()
        else:
            gids = list(group_ids)
        unique = sorted(set(gids))
        gid_to_idx = {g:i for i,g in enumerate(unique)}
        label_ids = torch.tensor([gid_to_idx[g] for g in gids], dtype=torch.long, device=device)
        N = features.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=device)
        sim = torch.matmul(features, features.T) / self.temperature
        mask_eye = torch.eye(N, device=device).bool()
        sim = sim.masked_fill(mask_eye, -9e15)
        labels_eq = label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
        labels_eq = labels_eq & (~mask_eye)
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1) + 1e-12
        numer = (exp_sim * labels_eq.float()).sum(dim=1)
        nonzero = (numer > 0).float()
        loss_per = -torch.log((numer / denom) + 1e-12) * nonzero
        n_pos_anchors = nonzero.sum()
        if n_pos_anchors.item() == 0:
            return torch.tensor(0.0, device=device)
        loss = loss_per.sum() / n_pos_anchors
        return loss


# ---------------- Training loop ----------------
def train_loop(
    index_map,
    model,
    device,
    out_dir,
    epochs=80,
    classes_per_batch=16,
    samples_per_class=4,
    temperature=0.07,
    lr=1e-3,
    weight_decay=1e-5,
    warmup_epochs=5,
    cosine=True,
    accum_steps=1,
    linear_probe_every=5,
    linear_probe_epochs=3,
    lead_indices=LEAD_INDICES,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_txt = out_dir / "training_report.txt"
    metrics_csv = out_dir / "metrics.csv"

    # write header
    with open(report_txt, "a") as fh:
        fh.write(f"Start training: {time.asctime()}\n")
    with open(metrics_csv, "w", newline="") as csvfh:
        writer = csv.writer(csvfh)
        writer.writerow(["epoch", "loss_supcon", "linear_probe_acc", "time_sec"])

    labels = [e["label"] for e in index_map]
    sampler = BalancedBatchSampler(labels, classes_per_batch, samples_per_class)
    dataset = SplitSegmentsDataset(index_map, lead_indices=lead_indices, augment='beat')
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=6, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=True)

    supcon = SupConLossTorch(temperature=temperature)
    best_metric = -1.0
    model.train()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_supcon = 0.0
        nb = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        opt.zero_grad()

        for batch in pbar:
            v1s, v2s, idxs = batch

            idxs = [int(i) for i in idxs]

            labs = [index_map[i]["label"] for i in idxs]

            # create repeated lists for augmented pairs (v1,v2)
            labs_rep = labs + labs

            # tensors
            v1 = torch.tensor(np.stack(v1s, axis=0), dtype=torch.float32).to(device)
            v2 = torch.tensor(np.stack(v2s, axis=0), dtype=torch.float32).to(device)

            x = torch.cat([v1, v2], dim=0)   # (2B, C, T)
            emb = model(x)
            emb = F.normalize(emb, dim=1)

            assert emb.size(0) == len(labs_rep)

            # positives = samples that share the same label (primary_scp)
            # assign a group id per disease label so all samples of same disease are positives
            lab_to_gid = {}
            group_ids = []
            next_gid = 0
            for lab in labs_rep:
                if lab not in lab_to_gid:
                    lab_to_gid[lab] = next_gid
                    next_gid += 1
                group_ids.append(lab_to_gid[lab])

            group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=emb.device)
            loss_sup = supcon(emb, group_ids_t)

            (loss_sup / max(1, 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            running_supcon += float(loss_sup.detach().cpu().item())
            nb += 1
            avg = running_supcon / nb if nb > 0 else 0.0
            pbar.set_postfix({"supcon": f"{avg:.6f}"})

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(running_supcon / max(1, nb))
        else:
            if epoch > warmup_epochs:
                scheduler.step()

        t_epoch = time.time() - t0
        lp_acc = -1.0
        if epoch % linear_probe_every == 0:
            try:
                lp_acc = train_linear_probe(model, index_map, lead_indices, device, epochs=linear_probe_epochs, lr=1e-3, batch_size=512, in_mem_limit=10000)
            except Exception:
                lp_acc = -1.0

        # write metrics
        with open(metrics_csv, "a", newline="") as csvfh:
            writer = csv.writer(csvfh)
            writer.writerow([epoch, running_supcon/nb if nb>0 else 0.0, lp_acc, t_epoch])
        with open(report_txt, "a") as fh:
            fh.write(f"Epoch {epoch} supcon={running_supcon/nb:.6f} lp_acc={lp_acc} time={t_epoch:.1f}s\n")
        print(f"[Epoch {epoch}] supcon={running_supcon/nb:.6f} lp_acc={lp_acc:.4f} time={t_epoch:.1f}s")

        # checkpoint (use lp_acc if valid)
        metric_score = lp_acc if lp_acc >= 0 else -(running_supcon/nb)
        if metric_score > best_metric:
            best_metric = metric_score
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim": opt.state_dict()}, out_dir / "best_encoder.pt")

    print("Training finished. Best metric:", best_metric)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ptbxl")
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument("--out", type=str, default="./contrastive_out")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_classes", type=int, default=16)
    parser.add_argument("--batch_samples", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--enc-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augment", type=str, choices=["basic","beat"], default="beat")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # collect shard/meta pairs for requested splits
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_pairs = []
    for sp in splits:
        pairs = find_shards_and_meta_for_split(args.dataset, sp)
        all_pairs.extend(pairs)

    index_map, class_counts = build_index_from_mapping(all_pairs, skip_null=True)
    print("Total usable segments:", len(index_map))
    print("Unique labels:", len(set(e["label"] for e in index_map)))
    print("Top classes:")
    for k, v in class_counts.most_common(10):
        print(f"  {k:20s} : {v}")

    # build encoder and start training
    encoder = ResNet1DEncoder(in_ch=len(LEAD_INDICES), hidden=args.enc_hidden, proj_dim=args.proj_dim, depth=4).to(device)
    train_loop(
        index_map,
        encoder,
        device,
        out_dir=args.out,
        epochs=args.epochs,
        classes_per_batch=args.batch_classes,
        samples_per_class=args.batch_samples,
        temperature=0.07,
        lr=args.lr,
        weight_decay=1e-5,
        warmup_epochs=5,
        cosine=True,
        accum_steps=1,
        linear_probe_every=5,
        linear_probe_epochs=3,
        lead_indices=LEAD_INDICES,
    )
