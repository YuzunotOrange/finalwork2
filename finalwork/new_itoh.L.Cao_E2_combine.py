#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#2025-10-18
#E2実装のためのプログラムnew_liフォルダ内すべてのデータを解析
#解析結果においてRQAとの食い違いが見られる

"""
Per-file Ito E2 comparator (with m* annotation)
- E2 は従来通り
- 追加: E1 を内部計算して m*（最小埋め込み次元）を推定
- 追加: E2 図に「名前: m*」の参照ボックス＆ m* の縦破線を描画
- 乱数やノイズは既定: seed=42, noise_type=uniform, noise_scale=1.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import glob
import shutil
import re

# 先頭のimport群の下あたりに追記
def minmax01(x):
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x); xmax = np.nanmax(x)
    rng = xmax - xmin
    if not np.isfinite(rng) or rng < 1e-12:
        # 定数系列のときはゼロ列を返す（分母ゼロ回避）
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / rng

def normalize_rqa_style(x):
    """RQAと同じ正規化: z-score"""
    x = np.asarray(x, dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / (sd if sd > 0 else 1.0)

def apply_norm(x, mode="none"):
    if mode == "minmax":
        return minmax01(x)
    elif mode == "zscore":
        s = np.std(x); return (x - np.mean(x)) / (s or 1.0)
    else:
        return np.asarray(x, dtype=float)


def sanitize_filename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^\w\-.]+", "_", base)
    return base or "output"

# ---------- utils: tau estimation ----------
def calculate_autocorrelation(series, max_lag: int):
    series = np.asarray(series, dtype=float)
    mu = np.mean(series)
    denom = np.sum((series - mu) ** 2)
    out = []
    for lag in range(1, max_lag + 1):
        if lag >= len(series):
            break
        num = np.sum((series[:-lag] - mu) * (series[lag:] - mu))
        out.append(num / denom if denom != 0 else 0.0)
    return out

def determine_tau(series, max_lag: int = 100):
    ac = calculate_autocorrelation(series, max_lag)
    thr = 1 / np.e
    for i, v in enumerate(ac):
        if v < thr:
            return i + 1
    return max_lag

# ---------- embedding ----------
def _embed(x, m, tau):
    x = np.asarray(x, dtype=float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

# ---------- Ito E1（m* 推定用に追加） ----------
def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if s is None:
        s = tau
    if k is None:
        k = max(1, int(0.05 * N))
    E1 = []
    for m in range(1, max_dim + 1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0:
            break
        valid_M = M - s
        X_now = Xm[:valid_M]
        k_eff = min(k, max(1, valid_M - 1))
        if k_eff < 1:
            break
        over_k = min(valid_M - 1, k_eff + 20)
        nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm='auto', metric="euclidean")
        nn.fit(X_now)
        dists, indxs = nn.kneighbors(X_now)
        ratios = []
        for i in range(valid_M):
            cand_j = indxs[i, 1:]
            cand_d = dists[i, 1:]
            if theiler > 0:
                mask = np.abs(cand_j - i) > theiler
                cand_j = cand_j[mask]; cand_d = cand_d[mask]
            if cand_j.size == 0:
                continue
            mask2 = (cand_j + s) < M
            cand_j = cand_j[mask2]; cand_d = cand_d[mask2]
            if cand_j.size == 0:
                continue
            use = min(k_eff, cand_j.size)
            jj = cand_j[:use]
            d_now_i = cand_d[:use].mean()
            if d_now_i < 1e-12:
                continue
            d_fut_i = np.linalg.norm(Xm[i + s] - Xm[jj + s], axis=1).mean()
            ratios.append(d_fut_i / d_now_i)
        E1.append(float(np.mean(ratios)) if ratios else np.nan)
    return np.array(E1, dtype=float)

def estimate_min_dimension(E1, eps=0.15, win=2):
    E1 = np.asarray(E1, dtype=float)
    if len(E1) == 0 or np.all(~np.isfinite(E1)):
        return 1
    for m in range(1, len(E1) - win + 1):
        seg = E1[m:m+win]
        dif = np.abs(np.diff(seg))
        base = np.maximum(np.abs(seg[:-1]), 1e-12)
        if np.all(dif / base <= eps):
            return m + 1
    return len(E1)

# ---------- Ito E2（従来通り） ----------
def _e_star(x, m, tau, k=None, theiler: int = 0):
    x = np.asarray(x, dtype=float)
    Xm = _embed(x, m, tau)
    M1 = Xm.shape[0]
    if M1 <= 0:
        return np.nan
    idx_last = np.arange(M1) + m * tau
    if np.any(idx_last >= len(x)):
        ok = np.where(idx_last < len(x))[0]
        if ok.size == 0:
            return np.nan
        Xm = Xm[: ok[-1] + 1]
        idx_last = idx_last[: ok[-1] + 1]
        M1 = Xm.shape[0]
        if M1 <= 1:
            return np.nan
    k_default = max(1, int(0.05 * M1))
    k_eff = max(1, min(k if k is not None else k_default, M1 - 1))
    over_k = min(M1 - 1, k_eff + 20)
    nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm="auto", metric="euclidean")
    nn.fit(Xm)
    _, index = nn.kneighbors(Xm)
    diffs = []
    for i in range(M1):
        cand = index[i, 1:]
        if theiler > 0:
            cand = cand[np.abs(cand - i) > theiler]
        if cand.size == 0:
            continue
        use = min(k_eff, cand.size)
        jj = cand[:use]
        xi_last = x[idx_last[i]]
        xj_last = x[idx_last[jj]]
        diffs.append(np.mean(np.abs(xi_last - xj_last)))
    return float(np.mean(diffs)) if diffs else np.nan

def itho_e2(x, max_dim: int = 10, tau: int = 5, k=None, theiler: int = 0):
    x = np.asarray(x, dtype=float)
    Estar = np.array([_e_star(x, m, tau, k, theiler) for m in range(1, max_dim + 1)], dtype=float)
    E2 = []
    for m in range(1, max_dim):
        a, b = Estar[m - 1], Estar[m]
        if not np.isfinite(a) or a == 0 or not np.isfinite(b):
            E2.append(np.nan)
        else:
            E2.append(b / a)
    return np.array(E2, dtype=float)

# ---------- I/O / Plot ----------
def load_series_whitespace_firstcol(path: str):
    df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    return df.iloc[:, 0].values

def plot_e2_curves(curves, savepath, max_dim=10, title="Ito E2 vs Embedding Dimension"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.hlines(y=1.0, xmin=1, xmax=max_dim - 1, linestyles="dashed", colors="gray",
              label="White Noise Reference (E2≈1)")
    ref_lines = []
    for c in curves:
        m = c["m"]; y = c["E2"]
        (ln,) = ax.plot(m, y, marker=c.get("marker", "o"), linestyle=c.get("linestyle", "-"),
                        linewidth=1.6, label=c["label"])
        # ここで m* を縦破線で表示（E1由来）
        if "m_star" in c and c["m_star"] is not None and 1 <= c["m_star"] <= max_dim - 1:
            ax.axvline(c["m_star"], color=ln.get_color(), linestyle="--", alpha=0.7)
        # 参照ボックスに「名前: m*」
        if "short_name" in c:
            ref_lines.append(f"{c['short_name']}: m*={c.get('m_star', 'NA')}")
    ax.set_xlabel("Embedding Dimension m")
    ax.set_ylabel("Ito E2 (= E*(m+1)/E*(m))")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.set_xticks(range(1, max_dim))
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(); plt.subplots_adjust(right=0.78)
    # 参照ボックス（左上）
    if ref_lines:
        ax.text(0.01, 0.99, "\n".join(ref_lines), transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))
    plt.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[Saved] {savepath}")

def compute_e2_for_series(x, max_dim, tau, k, theiler, label, marker, short_name):
    """E2に加えて E1->m* を同時算出して返す"""
    e2 = itho_e2(x, max_dim=max_dim, tau=tau, k=k, theiler=theiler)
    e1 = itho_e1(x, max_dim=max_dim, tau=tau, s=tau, k=k, theiler=theiler)
    m_star = int(estimate_min_dimension(e1, eps=0.15, win=2)) if e1.size > 0 else None
    return {
        "m": list(range(1, len(e2) + 1)),
        "E2": e2,
        "label": label,
        "marker": marker,
        "m_star": m_star,
        "short_name": short_name,  # 参照ボックス用に短い名前
    }

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Per-file Ito E2 (with m* annotation).")
    # argparse にオプション追加
    parser.add_argument("--norm", choices=["none", "minmax", "zscore"],
                    default="minmax", help="Normalization for E1/E2 (default=minmax)")
    parser.add_argument("--output_dir", default="new_Itho.E2_result", help="Directory to save outputs")
    parser.add_argument("--max_dim", type=int, default=10)
    parser.add_argument("--k_nn", type=int, default=None, help="fixed k (if omitted, use 5% per m internally)")
    parser.add_argument("--theiler", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default=42)")
    parser.add_argument("--noise_type", choices=["uniform", "gaussian"], default="uniform",
                        help="Noise type (default=uniform)")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Noise scale (default=1.0)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # fresh output dir
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Created: {args.output_dir}")

    # Lorenz (REQUIRED)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lorenz_path = os.path.join(script_dir, "lorenz.csv")
    if not os.path.exists(lorenz_path):
        raise FileNotFoundError("lorenz.csv not found (place it next to this script).")

    # Lorenz (REQUIRED)
    lorenz_raw = load_series_whitespace_firstcol(lorenz_path)
    lorenz = lorenz_raw            # ★ 正規化しない

    # ここで必ず初期化
    e2_curves_common = []

    # 素の Lorenz を追加
    e2_curves_common.append(
        compute_e2_for_series(
            lorenz, args.max_dim, tau=5, k=args.k_nn, theiler=args.theiler,
            label="Method 2 (lorenz, no noise, tau=5)", marker="s",
            short_name="Lorenz"
        )
    )

    # ノイズ付き Lorenz（必要なら）
    if args.noise_type == "uniform":
        noise = np.random.uniform(-args.noise_scale, args.noise_scale, size=lorenz_raw.shape)
    else:
        noise = np.random.normal(0, args.noise_scale, size=lorenz_raw.shape)
    lorenz_noisy = lorenz_raw + noise   # ★ これも正規化しない

    e2_curves_common.append(
        compute_e2_for_series(
            lorenz_noisy, args.max_dim, tau=5, k=args.k_nn, theiler=args.theiler,
            label=f"Method 2 (lorenz + {args.noise_type}, tau=5)", marker="^",
            short_name=f"Lorenz+{args.noise_type}"
        )
    )

    # Real data under new_li/*.csv
    data_dir = os.path.join(script_dir, "new_li")
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        print("[INFO] No files under new_li/. Saving Lorenz-only plot.")
        savepath = os.path.join(args.output_dir, "ito_e2_compare_lorenz_only.png")
        plot_e2_curves(e2_curves_common, savepath, max_dim=args.max_dim)
        print("\n[Done] Per-file Ito E2 comparison finished.")
        return

    for f in files:
        print(f"[INFO] Processing {f}")
        x_raw = load_series_whitespace_firstcol(f)
        x_std = (x_raw - np.mean(x_raw)) / (np.std(x_raw) or 1.0)
        tau1 = determine_tau(x_std, max_lag=100)
       
        x_norm = normalize_rqa_style(x_raw)
        curves=[]
        curves.append(
            compute_e2_for_series(
                x_norm, args.max_dim, tau=tau1, k=args.k_nn, theiler=args.theiler,
                label=f"Method 1 ({os.path.basename(f)}, tau={tau1})", marker="o",
                short_name=os.path.basename(f)
            )
        )

        
        print(f"[INFO] {os.path.basename(f)}: tau={tau1}; k policy = {'fixed '+str(args.k_nn) if args.k_nn else 'auto 5% per m'}")
         # ★ ここで Lorenz 系を重ねる
        curves.extend(e2_curves_common)
        
        # 実データ（短い表示名＝元ファイル名）
        short = os.path.basename(f)
        savepath = os.path.join(args.output_dir, f"ito_e2_compare_{sanitize_filename(f)}.png")
        plot_e2_curves(curves, savepath, max_dim=args.max_dim)

    print("\n[Done] Per-file Ito E2 comparison finished.")

if __name__ == "__main__":
    main()
