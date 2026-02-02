from pathlib import Path
import math
import gc
import json
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

ROOT = Path("./Cleaned_Datasets")
OUT_ROOT = Path("./Segments")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 100
SEGMENT_LENGTH = 256
HOP = 64
FINAL_SHARD_FLUSH = 2000
SAMPLE_FOR_THRESHOLDS = 12000

LOW_PCT = 0.1
HIGH_PCT = 99.9

DEFAULT_WORKERS = max(1, min(32, (mp.cpu_count() or 2) - 1))
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BAD_LEADS_MAX = 3
SAMPLES_FOR_PLOTTING = 100000


# helper
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# list files
def list_npy_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.npy") if p.is_file()])


# meta path
def meta_path_for_npy(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.stem + "_meta.json")


# json dump
def safe_json_dump(obj, path: Path):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
    except Exception:
        try:
            with open(path, "w") as fh:
                json.dump(obj, fh, indent=2)
        except Exception:
            pass


_RHYTHM_CODES = set(["AFIB", "AFLT", "STACH", "SBRAD", "PSVT", "SVTAC", "SR", "SB"])
_MORPHOLOGY_INDICATORS = set(
    ["ABQRS", "QWAVE", "LVOLT", "HVOLT", "LOWT", "LVH", "IVCD", "LBBB", "RBBB"]
)


# infer type
def infer_statement_type(record_meta: Dict) -> str:
    try:
        sc = record_meta.get("super_class")
        ps = record_meta.get("primary_scp")
        scp_keys = (
            list(record_meta.get("scp_codes", {}).keys())
            if record_meta.get("scp_codes")
            else []
        )
        if sc is not None and str(sc).strip() != "":
            return "diagnostic"
        if ps and str(ps).upper() in _RHYTHM_CODES:
            return "rhythm"
        if any(k.upper() in _MORPHOLOGY_INDICATORS for k in scp_keys):
            return "morphology"
        acrs = record_meta.get("diagnosis_acronyms", []) or []
        for a in acrs:
            if a.upper() in _RHYTHM_CODES:
                return "rhythm"
            if a.upper() in _MORPHOLOGY_INDICATORS:
                return "morphology"
        return "unknown"
    except Exception:
        return "unknown"


# read meta
def read_record_meta(npy_path: Path) -> Dict:
    mpth = meta_path_for_npy(npy_path)
    base_fallback = {
        "rec_id": npy_path.stem,
        "patient_id": npy_path.stem,
        "fs": None,
        "age": None,
        "sex": None,
        "source_path": str(npy_path),
        "scp_codes": {},
        "scp_codes_raw": None,
        "primary_scp": None,
        "super_class": None,
        "diagnostic_subclass": None,
    }
    if not mpth.exists():
        base_fallback["statement_type"] = infer_statement_type(base_fallback)
        return base_fallback
    try:
        with open(mpth, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        base_fallback["statement_type"] = infer_statement_type(base_fallback)
        return base_fallback
    out = {
        "rec_id": d.get("rec_id", base_fallback["rec_id"]),
        "patient_id": d.get("patient_id", base_fallback["patient_id"]),
        "fs": d.get("fs", base_fallback["fs"]),
        "age": d.get("age", base_fallback["age"]),
        "sex": d.get("sex", base_fallback["sex"]),
        "source_path": d.get("source_path", base_fallback["source_path"]),
        "scp_codes": d.get("scp_codes", {}) or {},
        "scp_codes_raw": d.get("scp_codes_raw", None),
        "primary_scp": d.get("primary_scp", None),
        "super_class": d.get("super_class", None),
        "diagnostic_subclass": d.get("diagnostic_subclass", None),
        "diagnosis_snomed_codes": d.get("diagnosis_snomed_codes", []) or [],
        "diagnosis_acronyms": d.get("diagnosis_acronyms", []) or [],
        "diagnosis_fullnames": d.get("diagnosis_fullnames", []) or [],
    }
    out["statement_type"] = infer_statement_type(out)
    return out


# ensure leads
def ensure_12_leads(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return None
    T, C = arr.shape
    if C < 12:
        pad = np.zeros((T, 12 - C), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    elif C > 12:
        arr = arr[:, :12]
    return arr


# pass1 stats
def sample_stats_from_file(args: Tuple[str, int, int, int]):
    npy_path_str, seg_len, hop, sample_per_file = args
    try:
        npy_path = Path(npy_path_str)
        arr = np.load(npy_path, mmap_mode="r")
        arr = ensure_12_leads(arr)
        if arr is None:
            return []
        T, C = arr.shape
        if T <= 0:
            return []
        starts = list(range(0, T, hop))
        valid_starts = [s for s in starts if (s + seg_len) <= T]
        if not valid_starts:
            return []
        if len(valid_starts) <= sample_per_file:
            chosen = valid_starts
        else:
            chosen = list(
                np.random.choice(valid_starts, size=sample_per_file, replace=False)
            )
        out = []
        for s in chosen:
            end = s + seg_len
            seg = arr[s:end, :]
            if not np.all(np.isfinite(seg)):
                continue
            amp = np.max(np.abs(seg), axis=0).astype(np.float32)
            rms = np.sqrt(np.mean(seg**2, axis=0)).astype(np.float32)
            out.append((amp, rms))
        del arr
        gc.collect()
        return out
    except Exception:
        return []


# pass2 extract
def extract_segments_sliding_from_file(args: Tuple[str, int, int]):
    npy_path_str, seg_len, hop = args
    try:
        npy_path = Path(npy_path_str)
        record_meta = read_record_meta(npy_path)
        arr = np.load(npy_path, mmap_mode="r")
        arr = ensure_12_leads(arr)
        if arr is None:
            return []
        T, C = arr.shape
        if T <= 0:
            return []
        segments_with_meta = []
        start = 0
        while (start + seg_len) <= T:
            end = start + seg_len
            seg = arr[start:end, :]
            if not np.all(np.isfinite(seg)):
                start += hop
                continue
            seg = seg.astype(np.float32, copy=False)
            seg_meta = {
                "rec_id": record_meta.get("rec_id"),
                "patient_id": record_meta.get("patient_id"),
                "source_path": record_meta.get("source_path"),
                "fs": record_meta.get("fs"),
                "age": record_meta.get("age"),
                "sex": record_meta.get("sex"),
                "start_sample": int(start),
                "end_sample": int(end),
                "scp_codes": record_meta.get("scp_codes", {}),
                "scp_codes_raw": record_meta.get("scp_codes_raw", None),
                "primary_scp": record_meta.get("primary_scp", None),
                "super_class": record_meta.get("super_class", None),
                "diagnostic_subclass": record_meta.get("diagnostic_subclass", None),
                "statement_type": record_meta.get(
                    "statement_type", infer_statement_type(record_meta)
                ),
                "diagnosis_snomed_codes": record_meta.get("diagnosis_snomed_codes", []),
                "diagnosis_acronyms": record_meta.get("diagnosis_acronyms", []),
                "diagnosis_fullnames": record_meta.get("diagnosis_fullnames", []),
            }
            segments_with_meta.append((seg, seg_meta))
            start += hop
        del arr
        gc.collect()
        return segments_with_meta
    except Exception:
        return []


# thresholds
def compute_percentile_thresholds_from_samples(
    amp_samples: List[np.ndarray],
    rms_samples: List[np.ndarray],
    low_pct=LOW_PCT,
    high_pct=HIGH_PCT,
):
    thresholds = {}
    if not amp_samples or not rms_samples:
        for li in range(12):
            thresholds[li] = {
                "amp_low": 0.0,
                "amp_high": float("inf"),
                "rms_low": 0.0,
                "rms_high": float("inf"),
            }
        return thresholds
    amps = np.vstack(amp_samples)
    rmss = np.vstack(rms_samples)
    for li in range(amps.shape[1]):
        a_col = amps[:, li]
        r_col = rmss[:, li]
        amp_low = float(np.percentile(a_col, low_pct))
        amp_high = float(np.percentile(a_col, high_pct))
        rms_low = float(np.percentile(r_col, low_pct))
        rms_high = float(np.percentile(r_col, high_pct))
        thresholds[li] = {
            "amp_low": amp_low,
            "amp_high": amp_high,
            "rms_low": rms_low,
            "rms_high": rms_high,
        }
    return thresholds


# report dirs
def make_report_dirs(dataset_name: str):
    base = OUT_ROOT / dataset_name
    reports = base / "reports"
    ensure_dir(reports)
    return base, reports


# plotting
def plot_hist_before_after(
    before_vals, after_vals, xlabel, outpath_before, outpath_after, bins=80
):
    try:
        plt.figure(figsize=(10, 4))
        plt.hist(before_vals, bins=bins)
        plt.title("Before (sampled) - " + xlabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(outpath_before, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.hist(after_vals, bins=bins)
        plt.title("After QC (sampled) - " + xlabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(outpath_after, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


# heatmap
def plot_per_lead_heatmap(counts_per_lead, title, outpath):
    if sns is None:
        return
    try:
        arr = np.array(counts_per_lead)
        plt.figure(figsize=(8, 2))
        sns.heatmap(arr[None, :], annot=True, fmt="d", cmap="Reds")
        plt.yticks([])
        plt.xticks(np.arange(12) + 0.5, [f"L{i+1}" for i in range(12)])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


# boxplot
def plot_boxplot(values, ylabel, outpath):
    try:
        plt.figure(figsize=(6, 4))
        plt.boxplot(values)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


# two-pass
def process_dataset_two_passes(
    dataset_name: str,
    dataset_folder: Path,
    workers: int = DEFAULT_WORKERS,
    seg_len: int = SEGMENT_LENGTH,
    hop: int = HOP,
    sample_target: int = SAMPLE_FOR_THRESHOLDS,
    shard_flush: int = FINAL_SHARD_FLUSH,
    bad_leads_max: int = BAD_LEADS_MAX,
    samples_for_plotting: int = SAMPLES_FOR_PLOTTING,
):
    print(f"\n=== Processing dataset: {dataset_name}  (folder: {dataset_folder}) ===")
    files = list_npy_files(dataset_folder)
    print(f"Found {len(files)} .npy files in {dataset_folder}")
    if len(files) == 0:
        print("No files, skipping.")
        return

    per_file_k = max(1, int(math.ceil(sample_target / max(1, len(files)))))
    amp_samples = []
    rms_samples = []

    args = [(str(p), seg_len, hop, per_file_k) for p in files]
    with mp.Pool(processes=workers) as pool:
        for file_samples in tqdm(
            pool.imap(sample_stats_from_file, args),
            total=len(args),
            desc="Sampling files",
        ):
            if not file_samples:
                continue
            for amp, rms in file_samples:
                amp_samples.append(amp)
                rms_samples.append(rms)
                if len(amp_samples) >= sample_target:
                    break
            if len(amp_samples) >= sample_target:
                break

    print(f"Collected sample_count={len(amp_samples)} for threshold estimation")
    thresholds = compute_percentile_thresholds_from_samples(amp_samples, rms_samples)

    base, reports_dir = make_report_dirs(dataset_name)
    try:
        with open(base / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
    except Exception:
        pass

    rejected_reason_counter = Counter()
    rejected_per_lead = [0] * 12
    accepted_per_lead_counts = [0] * 12

    before_amp_samples_flat = []
    before_rms_samples_flat = []
    after_amp_samples_flat = []
    after_rms_samples_flat = []

    all_dir = base / "all"
    ensure_dir(all_dir)
    buffer_segments = []
    buffer_meta = []
    shard_idx = 0
    total_seen = 0
    accepted = 0
    rejected = 0

    primary_counter = Counter()
    super_counter = Counter()

    args2 = [(str(p), seg_len, hop) for p in files]
    with mp.Pool(processes=workers) as pool:
        for segments_with_meta in tqdm(
            pool.imap(extract_segments_sliding_from_file, args2),
            total=len(args2),
            desc="Sliding-window extract (pass2)",
        ):
            if not segments_with_meta:
                continue
            for seg, seg_meta in segments_with_meta:
                total_seen += 1
                if seg is None or seg.size == 0 or not np.all(np.isfinite(seg)):
                    rejected += 1
                    rejected_reason_counter["nonfinite_or_empty"] += 1
                    continue
                amp = np.max(np.abs(seg), axis=0)
                rms = np.sqrt(np.mean(seg**2, axis=0))

                if len(before_amp_samples_flat) < samples_for_plotting:
                    before_amp_samples_flat.extend(list(amp))
                if len(before_rms_samples_flat) < samples_for_plotting:
                    before_rms_samples_flat.extend(list(rms))

                per_lead_flags = [False] * 12
                per_lead_reasons = []
                for li in range(12):
                    thr = thresholds.get(li, None)
                    if thr is None:
                        continue
                    a = float(amp[li])
                    r = float(rms[li])
                    reason_keys = []
                    if a > thr["amp_high"]:
                        reason_keys.append("amp_high")
                    if a < thr["amp_low"]:
                        reason_keys.append("amp_low")
                    if r > thr["rms_high"]:
                        reason_keys.append("rms_high")
                    if r < thr["rms_low"]:
                        reason_keys.append("rms_low")
                    if reason_keys:
                        per_lead_flags[li] = True
                        per_lead_reasons.append((li, reason_keys))

                bad_leads = sum(1 for f in per_lead_flags if f)

                if bad_leads > bad_leads_max:
                    rejected += 1
                    rejected_reason_counter["too_many_bad_leads"] += 1
                    for li, flag in enumerate(per_lead_flags):
                        if flag:
                            rejected_per_lead[li] += 1
                    continue

                buffer_segments.append(seg.astype(np.float32))
                seg_meta_qc = dict(seg_meta)
                seg_meta_qc.update(
                    {
                        "qc_rejected": False,
                        "qc_bad_leads_count": int(bad_leads),
                        "qc_per_lead_flags": per_lead_flags,
                        "qc_per_lead_reasons": {
                            str(li): reasons for (li, reasons) in per_lead_reasons
                        },
                    }
                )
                buffer_meta.append(seg_meta_qc)

                accepted += 1

                for li in range(12):
                    if not per_lead_flags[li]:
                        accepted_per_lead_counts[li] += 1

                if len(after_amp_samples_flat) < samples_for_plotting:
                    after_amp_samples_flat.extend(list(amp))
                if len(after_rms_samples_flat) < samples_for_plotting:
                    after_rms_samples_flat.extend(list(rms))

                ps = seg_meta.get("primary_scp")
                sc = seg_meta.get("super_class")
                if ps:
                    primary_counter[ps] += 1
                if sc:
                    super_counter[sc] += 1

                if len(buffer_segments) >= shard_flush:
                    out_path = all_dir / f"all_shard_{shard_idx:04d}.npy"
                    np.save(out_path, np.array(buffer_segments, dtype=np.float32))
                    meta_out_path = all_dir / f"all_shard_{shard_idx:04d}_meta.json"
                    safe_json_dump(buffer_meta, meta_out_path)
                    shard_idx += 1
                    buffer_segments.clear()
                    buffer_meta.clear()
                    gc.collect()

    if buffer_segments:
        out_path = all_dir / f"all_shard_{shard_idx:04d}.npy"
        np.save(out_path, np.array(buffer_segments, dtype=np.float32))
        meta_out_path = all_dir / f"all_shard_{shard_idx:04d}_meta.json"
        safe_json_dump(buffer_meta, meta_out_path)
        shard_idx += 1
        buffer_segments.clear()
        buffer_meta.clear()
        gc.collect()

    summary = {
        "shards_written": shard_idx,
        "accepted_segments": accepted,
        "rejected_segments": rejected,
        "total_seen_segments": total_seen,
        "rejected_reason_counts": dict(rejected_reason_counter.most_common()),
        "rejected_per_lead": rejected_per_lead,
        "accepted_per_lead_counts": accepted_per_lead_counts,
        "primary_scp_counts": dict(primary_counter.most_common()),
        "super_class_counts": dict(super_counter.most_common()),
    }
    safe_json_dump(summary, base / "shard_summary.json")

    print(
        f"Finished dataset {dataset_name}: shards_written={shard_idx}, accepted={accepted}, rejected={rejected}, total_seen_segments={total_seen}"
    )

    try:
        before_amp = (
            np.array(before_amp_samples_flat) if before_amp_samples_flat else np.array([])
        )
        before_rms = (
            np.array(before_rms_samples_flat) if before_rms_samples_flat else np.array([])
        )
        after_amp = (
            np.array(after_amp_samples_flat) if after_amp_samples_flat else np.array([])
        )
        after_rms = (
            np.array(after_rms_samples_flat) if after_rms_samples_flat else np.array([])
        )

        if before_amp.size > 0 and after_amp.size > 0:
            plot_hist_before_after(
                before_amp,
                after_amp,
                "Max abs amplitude (flattened)",
                reports_dir / "amp_before.png",
                reports_dir / "amp_after.png",
            )
        if before_rms.size > 0 and after_rms.size > 0:
            plot_hist_before_after(
                before_rms,
                after_rms,
                "RMS (flattened)",
                reports_dir / "rms_before.png",
                reports_dir / "rms_after.png",
            )

        plot_per_lead_heatmap(
            rejected_per_lead,
            "Rejected segments per lead",
            reports_dir / "rejected_per_lead_heatmap.png",
        )

        if before_amp.size > 0:
            plot_boxplot(before_amp, "amp_before", reports_dir / "amp_before_box.png")
        if after_amp.size > 0:
            plot_boxplot(after_amp, "amp_after", reports_dir / "amp_after_box.png")

        try:
            keys = list(summary["rejected_reason_counts"].keys())
            vals = list(summary["rejected_reason_counts"].values())
            if keys:
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(vals)), vals)
                plt.xticks(range(len(vals)), keys, rotation=45, ha="right")
                plt.title("Rejected segments by reason")
                plt.tight_layout()
                plt.savefig(reports_dir / "rejection_reasons_bar.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(6, 6))
                plt.pie(vals, labels=keys, autopct="%1.1f%%")
                plt.title("Rejection reasons pie")
                plt.savefig(reports_dir / "rejection_reasons_pie.png", bbox_inches="tight")
                plt.close()
        except Exception:
            pass

        try:
            accept_rates = []
            for li in range(12):
                tot = accepted_per_lead_counts[li] + rejected_per_lead[li]
                rate = (accepted_per_lead_counts[li] / tot) if tot > 0 else 0.0
                accept_rates.append(rate)
            plt.figure(figsize=(8, 3))
            plt.bar(range(12), accept_rates)
            plt.xticks(range(12), [f"L{i+1}" for i in range(12)])
            plt.ylim(0, 1.0)
            plt.title("Per-lead accept rate (accepted / (accepted+rejected))")
            plt.tight_layout()
            plt.savefig(reports_dir / "per_lead_accept_rate.png", bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        try:
            import pandas as pd

            df_summary = pd.DataFrame(
                {
                    "lead": [f"L{i+1}" for i in range(12)],
                    "rejected_count": rejected_per_lead,
                    "accepted_count": accepted_per_lead_counts,
                }
            )
            df_summary.to_csv(reports_dir / "per_lead_summary.csv", index=False)
        except Exception:
            pass

        try:
            if thresholds:
                amp_lows = [thresholds[i]["amp_low"] for i in range(12)]
                amp_highs = [thresholds[i]["amp_high"] for i in range(12)]
                plt.figure(figsize=(8, 3))
                plt.errorbar(
                    range(12),
                    [(l + h) / 2.0 for l, h in zip(amp_lows, amp_highs)],
                    yerr=[(h - l) / 2.0 for l, h in zip(amp_lows, amp_highs)],
                    fmt="o",
                )
                plt.xticks(range(12), [f"L{i+1}" for i in range(12)])
                plt.title("Per-lead amplitude thresholds (low/high)")
                plt.tight_layout()
                plt.savefig(reports_dir / "amp_thresholds.png", bbox_inches="tight")
                plt.close()
        except Exception:
            pass

    except Exception:
        pass

    print(f"Reports and shard summary saved to: {reports_dir.resolve()}")


if __name__ == "__main__":
    ptbxl_folder = ROOT / "ptbxl_clean"
    if not ptbxl_folder.exists():
        print("No PTB-XL cleaned folder found under Cleaned_Datasets. Exiting.")
        raise SystemExit(0)
    process_dataset_two_passes("ptbxl", ptbxl_folder, workers=DEFAULT_WORKERS)
    print("\nDone. Segmented shards + QC reports under ./Segments/<dataset>/")
