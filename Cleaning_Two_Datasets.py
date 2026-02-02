from pathlib import Path
import os
import json
import gc
import argparse
import ast
import re
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import wfdb
except Exception:
    wfdb = None
from scipy.signal import butter, filtfilt, medfilt, decimate, resample_poly, iirnotch

TARGET_LENGTH = 1000
OUT_ROOT = Path("./Cleaned_Datasets").resolve()
REPORTS_ROOT = OUT_ROOT / "reports"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

PTBXL_PATH = Path("../../ptbxl")
DATABASE_CSV = PTBXL_PATH / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = PTBXL_PATH / "scp_statements.csv"
PTBXL_OUT_DIR = OUT_ROOT / "ptbxl_clean"
PTBXL_OUT_DIR.mkdir(parents=True, exist_ok=True)
PTBXL_FS = 100.0
PTBXL_USE_LR = True

POWERLINE_FREQS = (50.0, 60.0)
NOTCH_Q = 30.0
DEFAULT_WORKERS = max(1, min(32, (multiprocessing.cpu_count() or 2) - 1))


# helper
def _choose_powerline_freq(fs: float):
    nyq = fs / 2.0
    freqs = [f for f in POWERLINE_FREQS if f < nyq]
    if not freqs:
        return None
    return freqs[0] if len(freqs) == 1 else (50.0 if 50.0 < nyq else 60.0)


# filter
def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 0.9999
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0, padlen=3 * (max(len(a), len(b))))


# filter
def notch_filter(data: np.ndarray, fs: float, freq: float = None, q: float = NOTCH_Q):
    if freq is None:
        freq = _choose_powerline_freq(fs)
    if freq is None:
        return data
    nyq = fs / 2.0
    w0 = freq / nyq
    if not (0 < w0 < 1.0):
        return data
    b, a = iirnotch(w0, q)
    try:
        return filtfilt(b, a, data, axis=0, padlen=3 * (max(len(a), len(b))))
    except Exception:
        out = np.zeros_like(data)
        for ch in range(data.shape[1]):
            out[:, ch] = filtfilt(b, a, data[:, ch], padlen=3 * (max(len(a), len(b))))
        return out


# filter
def baseline_wander_removal(data: np.ndarray, fs: float = 100.0, window_sec: float = 0.6):
    window_size = int(round(window_sec * fs))
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    baseline = np.zeros_like(data)
    for ch in range(data.shape[1]):
        baseline[:, ch] = medfilt(data[:, ch], kernel_size=window_size)
    return data - baseline


# pad
def pad_or_trim(ecg: np.ndarray, targ_len: int = TARGET_LENGTH) -> np.ndarray:
    if ecg is None:
        return np.zeros((targ_len, 12), dtype=np.float32)
    t, c = ecg.shape
    if t > targ_len:
        return ecg[:targ_len, :]
    elif t < targ_len:
        pad = np.zeros((targ_len - t, c), dtype=ecg.dtype)
        return np.concatenate([ecg, pad], axis=0)
    return ecg


# resample
def downsample_to(arr: np.ndarray, orig_fs: float, new_fs: float) -> np.ndarray:
    if new_fs <= 0 or orig_fs <= 0:
        raise ValueError("Sampling rates must be positive")
    if abs(new_fs - orig_fs) < 1e-6:
        return arr
    factor = orig_fs / new_fs
    if abs(round(factor) - factor) < 1e-6 and round(factor) >= 1:
        q = int(round(factor))
        out_list = []
        for ch in range(arr.shape[1]):
            try:
                out_ch = decimate(arr[:, ch], q, ftype="fir", zero_phase=True)
            except TypeError:
                out_ch = decimate(arr[:, ch], q, ftype="fir")
            out_list.append(out_ch)
        out = np.vstack([o for o in out_list]).T
        return out
    else:
        from fractions import Fraction
        frac = Fraction(new_fs / orig_fs).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        out = resample_poly(arr, up, down, axis=0)
        return out


# io
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


# scp map
def load_scp_statements_map(scp_csv: Path):
    mapping = {}
    if not scp_csv.exists():
        return mapping
    try:
        df = pd.read_csv(scp_csv, dtype=str).fillna("")
        code_col = None
        super_col = None
        for c in df.columns:
            lc = c.lower()
            if "code" in lc or "scp" in lc:
                code_col = c
            if "super" in lc or "superclass" in lc or "super_class" in lc:
                super_col = c
        if code_col is None and len(df.columns) >= 1:
            code_col = df.columns[0]
        if super_col is None:
            for c in df.columns:
                if df[c].nunique() < 500:
                    super_col = c
                    break
        if code_col is None:
            return mapping
        for _, r in df.iterrows():
            code = str(r.get(code_col, "")).strip()
            sc = str(r.get(super_col, "")).strip() if super_col else ""
            if code:
                mapping[code] = sc if sc else ""
    except Exception:
        pass
    return mapping


SCP_TO_SUPER = load_scp_statements_map(SCP_STATEMENTS_CSV) if SCP_STATEMENTS_CSV.exists() else {}


# reader
def _read_ptbxl_record_local(filename: Path):
    if wfdb is None:
        raise RuntimeError("wfdb not available to read PTB-XL local records")
    record = wfdb.rdrecord(str(filename))
    sig = record.p_signal.astype(np.float32)
    if sig.shape[1] >= 12:
        sig = sig[:, :12]
    else:
        pad = np.zeros((sig.shape[0], 12 - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)
    return sig


# process
def _process_ptbxl_worker(args):
    idx, fn_path, out_dir, meta_info, keep_noisy = args
    try:
        rec_id = f"ptbxl_{idx:07d}"
        sig = _read_ptbxl_record_local(Path(fn_path))
        sig_clean = notch_filter(sig, fs=PTBXL_FS)
        sig_clean = butter_bandpass_filter(sig_clean, fs=PTBXL_FS)
        sig_clean = baseline_wander_removal(sig_clean, fs=PTBXL_FS)
        sig_clean = pad_or_trim(sig_clean, targ_len=TARGET_LENGTH)
        out_path = Path(out_dir) / f"{rec_id}.npy"
        np.save(out_path, sig_clean.astype(np.float32))
        meta = {
            "rec_id": rec_id,
            "dataset": "ptbxl",
            "source_path": str(fn_path),
            "patient_id": None,
            "age": None,
            "sex": None,
            "fs": PTBXL_FS,
        }
        try:
            if isinstance(meta_info, dict):
                meta["patient_id"] = meta_info.get("patient_id")
                meta["age"] = meta_info.get("age")
                meta["sex"] = meta_info.get("sex")
                if meta_info.get("scp_codes_raw"):
                    meta["scp_codes_raw"] = meta_info.get("scp_codes_raw")
                if meta_info.get("primary_scp"):
                    meta["primary_scp"] = meta_info.get("primary_scp")
                if meta_info.get("super_class"):
                    meta["super_class"] = meta_info.get("super_class")
        except Exception:
            pass
        meta_path = Path(out_dir) / f"{rec_id}_meta.json"
        safe_json_dump(meta, meta_path)
        del sig, sig_clean
        gc.collect()
        return {"status": "kept", "rec_id": rec_id}
    except Exception as e:
        return {"status": "error", "reason": str(e), "rec_id": None}


# report
def summarize_and_report(dataset_name: str, out_dir: Path, results: list):
    reports_dir = REPORTS_ROOT / dataset_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    total = len(results)
    kept = sum(1 for r in results if r and r.get("status") == "kept")
    errors = sum(1 for r in results if r and r.get("status") == "error")
    reasons = {}
    for r in results:
        if r and r.get("status") == "error":
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
    summary = {
        "dataset": dataset_name,
        "total_records": total,
        "kept": kept,
        "errors": errors,
        "rejection_reasons": reasons,
    }
    safe_json_dump(summary, reports_dir / f"{dataset_name}_summary.json")
    try:
        pd.DataFrame(results).to_csv(reports_dir / f"{dataset_name}_records_summary.csv", index=False)
    except Exception:
        pass
    print(f"Reports saved to: {reports_dir}")


# runner
def run_ptbxl(keep_noisy: bool, workers: int = DEFAULT_WORKERS):
    print("\n=== PTB-XL: Loading metadata and enumerating records ===")
    if not DATABASE_CSV.exists():
        print("[ERROR] PTBXL CSV not found at", DATABASE_CSV)
        return
    df = pd.read_csv(DATABASE_CSV)
    expected_cols = ["validated_by_human", "filename_lr", "filename_hr", "electrodes_problems", "pacemaker", "burst_noise", "static_noise", "patient_id", "age", "sex", "scp_codes"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    if keep_noisy:
        df_sel = df[df["filename_lr"].notnull() if PTBXL_USE_LR else df["filename_hr"].notnull()]
    else:
        df_sel = df[(df.get("validated_by_human", False) == True) & (df["filename_lr"].notnull())]
        df_sel = df_sel[(df_sel["electrodes_problems"] == 0) & (df_sel["pacemaker"] == 0) & (df_sel["burst_noise"] == 0) & (df_sel["static_noise"] == 0)]
    file_paths = []
    meta_infos = []
    for i, (_, row) in enumerate(df_sel.iterrows()):
        rel_path = row["filename_lr"] if PTBXL_USE_LR else row["filename_hr"]
        full_path = PTBXL_PATH / rel_path
        file_paths.append(str(full_path))
        scp_raw = row.get("scp_codes", None)
        scp_dict = {}
        primary_scp = None
        super_class = None
        if isinstance(scp_raw, str) and scp_raw.strip():
            try:
                parsed = ast.literal_eval(scp_raw)
                if isinstance(parsed, dict) and parsed:
                    scp_dict = parsed
                    primary_scp = max(parsed, key=parsed.get)
                    super_class = SCP_TO_SUPER.get(primary_scp, None)
            except Exception:
                toks = re.findall(r"[A-Z0-9_]{2,10}", str(scp_raw))
                if toks:
                    primary_scp = toks[0]
                    super_class = SCP_TO_SUPER.get(primary_scp, None)
        meta_infos.append({
            "patient_id": (None if pd.isna(row.get("patient_id")) else row.get("patient_id")),
            "age": None if pd.isna(row.get("age")) else row.get("age"),
            "sex": None if pd.isna(row.get("sex")) else row.get("sex"),
            "scp_codes_raw": scp_dict,
            "primary_scp": primary_scp,
            "super_class": super_class,
        })
    print(f"PTB-XL: {len(file_paths)} records to process (workers={workers})")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        futures = []
        for i, fp in enumerate(file_paths):
            futures.append(exe.submit(_process_ptbxl_worker, (i, fp, str(PTBXL_OUT_DIR), meta_infos[i], keep_noisy)))
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="PTB-XL"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"status": "error", "reason": str(e)})
    summarize_and_report("ptbxl", PTBXL_OUT_DIR, results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl", action="store_true")
    p.add_argument("--keep-noisy", dest="keep_noisy", action="store_true")
    p.add_argument("--no-keep-noisy", dest="keep_noisy", action="store_false")
    p.set_defaults(keep_noisy=True)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()
    if not args.ptbxl:
        print("Nothing requested. Use --ptbxl")
        raise SystemExit(0)
    run_ptbxl(keep_noisy=args.keep_noisy, workers=args.workers)
