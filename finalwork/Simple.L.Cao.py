#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Cao's method (E1/E2, Chebyshev distance per L. Cao 1997)
- Scan ./new_li/*.csv
- For each file, compare with lorenz.csv and lorenz.csv + white noise in one plot
- Save figures and summary.csv into ./L.Cao_result (recreated each run)

Usage:
  python cao_batch.py
Options:
  --input_dir   ./new_li
  --lorenz      ./lorenz.csv
  --max_m       10
  --tau         auto  (int or "auto": 1/e 自己相関)
  --noise_level 0.05  (lorenz に加える白色雑音の標準偏差 = data_std * noise_level)
  --output_dir  ./L.Cao_result
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Basic I/O --------------------

def read_series_first_col(path: Path) -> np.ndarray:
    """
    1列目のみを安全に読み取る（CSV/TSV/空白区切り・コメント行対応）
    """
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            s = s.replace(",", " ").replace("\t", " ")
            t = s.split()
            try:
                v = float(t[0])
            except Exception:
                continue
            vals.append(v)
    if not vals:
        raise RuntimeError(f"No numeric data found in first column: {path}")
    return np.asarray(vals, dtype=float)

def ensure_clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# -------------------- Tau helper --------------------

def autocorr_tau_1_over_e(x: np.ndarray, max_lag: int = 100) -> int:
    """
    自己相関が 1/e を初めて下回るラグを tau とする（簡易）
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return 1
    if max_lag is None:
        max_lag = max(10, len(x)//4)  # データ長の1/4くらいまで見る
    L = min(max_lag, len(x) - 1)
    for lag in range(1, L + 1):
        num = np.dot(x[:-lag], x[lag:])
        r = num / denom
        if r < 1 / np.e:
            return lag
    return max(1, L)

# -------------------- Cao (E1/E2) --------------------
def embed_trajectory(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """遅れ座標再構成（形状: [N_eff, m]）。作れない場合は空を返す。"""
    N = len(x)
    N_eff = N - (m - 1) * tau
    if N_eff <= 1:
        return np.empty((0, m))
    # 各列が遅れ次元
    return np.stack([x[i:i+N_eff] for i in range(0, m * tau, tau)], axis=1)

def chebyshev_nn(X: np.ndarray) -> np.ndarray:
    """
    各点の最近傍インデックス（最大ノルム）を返す。O(N^2)の単純実装。
    X: (N_eff, m)
    戻り値: nn (N_eff,), 各 i について i!=nn[i]
    """
    N = X.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=int)
    # 距離行列（Chebyshev: ∞ノルム）
    # dist[i,j] = max_k |X[i,k]-X[j,k]|
    dist = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)  # (N,N)
    np.fill_diagonal(dist, np.inf)  # 自己は除外
    nn = np.argmin(dist, axis=1)
    # 念のため境界チェック
    nn = np.clip(nn, 0, N - 1)
    return nn

def cao_E1_E2(x: np.ndarray, tau: int, max_m: int):
    """
    Cao (1997) の E1(m), E2(m) を m=1..usable_max_m で算出（最大ノルム）。
    系列長に応じて usable_max_m を自動調整。
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if tau <= 0:
        tau = 1

    # m と m+1 の両方で埋め込みが必要 → N - m*tau > 1
    usable_max_m = min(max_m, max(0, (N - 2) // tau))
    if usable_max_m < 1:
        return np.array([]), np.array([])

    # m=1..(usable_max_m+1) を準備
    embeds = [embed_trajectory(x, m, tau) for m in range(1, usable_max_m + 2)]

    E1 = np.zeros(usable_max_m)
    E2 = np.zeros(usable_max_m)

    for m in range(1, usable_max_m + 1):
        Xm  = embeds[m - 1]  # [N_eff_m, m]
        Xm1 = embeds[m]      # [N_eff_{m+1}, m+1]
        # m+1 に合わせる
        N_eff = Xm1.shape[0]
        if N_eff < 2:
            continue
        Xm = Xm[:N_eff]

        # 最近傍（最大ノルム）
        nn = chebyshev_nn(Xm)

        d_m  = np.max(np.abs(Xm  - Xm[nn]),  axis=1)
        d_m1 = np.max(np.abs(Xm1 - Xm1[nn]), axis=1)

        d_m = np.clip(d_m, 1e-8, None)      # 0 除算防止
        E1[m - 1] = np.mean(d_m1 / d_m)

        # E2
        scalar_m,  scalar_mn  = Xm[:, -1],  Xm[nn, -1]
        scalar_m1, scalar_m1n = Xm1[:, -1], Xm1[nn, -1]
        Ep_m  = np.mean(np.abs(scalar_m  - scalar_mn))
        Ep_m1 = np.mean(np.abs(scalar_m1 - scalar_m1n))
        Ep_m = max(Ep_m, 1e-8)
        E2[m - 1] = Ep_m1 / Ep_m

    return E1, E2

def estimate_m0(E1: np.ndarray, tol: float = 0.02, k: int = 1) -> int:
    """
    E1 の収束から最小埋め込み次元を推定：
    |E1(m+1)-E1(m)|<tol が k 回連続する最初の m を返す（見つからなければ 0）。
    """
    if len(E1) < 2:
        return 0
    diffs = np.abs(np.diff(E1))
    run = 0
    for m in range(len(diffs)):
        if diffs[m] < tol:
            run += 1
            if run >= k:
                return m + 1
        else:
            run = 0
    return 0

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",   default="./new_li")
    ap.add_argument("--lorenz",      default="./lorenz.csv")
    ap.add_argument("--max_m",       type=int, default=10)
    ap.add_argument("--tau",         default="auto", help='int or "auto" (1/e autocorr)')
    ap.add_argument("--noise_level", type=float, default=0.05)
    ap.add_argument("--output_dir",  default="./L.Cao_result")
    ap.add_argument("--tol",         type=float, default=0.02)
    ap.add_argument("--k",           type=int, default=1)
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] input_dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir = ensure_clean_dir(Path(args.output_dir))

    # --- Lorenz baseline & noisy ---
    lorenz = read_series_first_col(Path(args.lorenz))
    tau_lorenz = autocorr_tau_1_over_e(lorenz) if str(args.tau).lower()=="auto" else int(args.tau)
    E1_lz, E2_lz = cao_E1_E2(lorenz, tau=tau_lorenz, max_m=args.max_m)
    m0_lz = estimate_m0(E1_lz, tol=args.tol, k=args.k)

    rng = np.random.default_rng(0)
    noisy = lorenz + rng.normal(0.0, np.std(lorenz) * args.noise_level, size=len(lorenz))
    tau_noisy = autocorr_tau_1_over_e(noisy) if str(args.tau).lower()=="auto" else int(args.tau)
    E1_nz, E2_nz = cao_E1_E2(noisy, tau=tau_noisy, max_m=args.max_m)
    m0_nz = estimate_m0(E1_nz, tol=args.tol, k=args.k)

    # --- Target files ---
    files = sorted([p for p in input_dir.glob("*.csv") if p.resolve()!=Path(args.lorenz).resolve()])

    rows = []
    header = ["file","N","tau","m0",
              "lorenz_tau","lorenz_m0",
              "lorenz_noisy_tau","lorenz_noisy_m0"]

    for path in files:
        try:
            x = read_series_first_col(path)
        except Exception as e:
            print(f"[skip] {path.name}: {e}", file=sys.stderr)
            continue

        tau = autocorr_tau_1_over_e(x) if str(args.tau).lower()=="auto" else int(args.tau)
        E1, E2 = cao_E1_E2(x, tau=tau, max_m=args.max_m)
        if len(E1)==0:
            print(f"[skip] {path.name}: too short for max_m={args.max_m}, tau={tau}", file=sys.stderr)
            continue
        m0 = estimate_m0(E1, tol=args.tol, k=args.k)

        # ---- Plot per file ----
        plt.figure(figsize=(10,5))
        xs = np.arange(1, len(E1)+1)
        plt.plot(xs, E1, label=f"{path.name} (tau={tau}, m0={m0 if m0>0 else 'n/a'})")

        if len(E1_lz)>0:
            plt.plot(np.arange(1,len(E1_lz)+1), E1_lz, label=f"lorenz (tau={tau_lorenz}, m0={m0_lz if m0_lz>0 else 'n/a'})")
        if len(E1_nz)>0:
            plt.plot(np.arange(1,len(E1_nz)+1), E1_nz, label=f"lorenz+noise (tau={tau_noisy}, m0={m0_nz if m0_nz>0 else 'n/a'})")

        if m0>0:     plt.axvline(m0,      ls="--", color="gray", alpha=0.7)
        if m0_lz>0:  plt.axvline(m0_lz,   ls="--", color="gray", alpha=0.5)
        if m0_nz>0:  plt.axvline(m0_nz,   ls="--", color="gray", alpha=0.5)

        plt.xlabel("Embedding dimension m")
        plt.ylabel("E1 value")
        plt.yscale('log')
        plt.title(f"Cao E1 comparison: {path.name}")
        plt.grid(True, alpha=0.4)
        # 重複回避のためユニーク化
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if uniq:
            plt.legend(uniq.values(), uniq.keys())
        plt.tight_layout()
        out_png = output_dir / f"{path.stem}_cao.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[saved] {out_png}")

        rows.append([path.name, len(x), tau, m0, tau_lorenz, m0_lz, tau_noisy, m0_nz])

    if rows:
        df = pd.DataFrame(rows, columns=header)
        df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
        print(f"[saved] {output_dir / 'summary.csv'}")

    # console reference
    print("---- Lorenz reference ----")
    print(f"lorenz: N={len(lorenz)}, tau={tau_lorenz}, m0={m0_lz}")
    print(f"lorenz+noise: tau={tau_noisy}, m0={m0_nz}, noise_level={args.noise_level}")

if __name__ == "__main__":
    main()
