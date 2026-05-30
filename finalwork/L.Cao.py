#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L.Cao's method (E1(d), E2(d)) for estimating minimum embedding dimension.
Usage:
  python L.Cao.py --csv_files lorenz.csv
  python L.Cao.py --csv_files a.csv b.csv --column x --tau 1 --d_max 12 --theiler 0 --normalize zscore --save_prefix out
"""

import argparse
import sys
import math
from typing import Tuple, Optional, Sequence
import numpy as np
import pandas as pd

# -------------------------
# Core: Cao's E1, E2
# -------------------------
def _embed_trajectory(x: np.ndarray, d: int, tau: int) -> np.ndarray:
    """Create delay-coordinate vectors X^{(d)} with shape (m, d)."""
    m = x.shape[0] - d * tau
    if m <= 0:
        return np.empty((0, d))
    # Column-major stacking for speed; equivalent to sliding window with step tau
    return np.column_stack([x[k * tau : k * tau + m] for k in range(d)])

def _nearest_neighbors_maxnorm(X: np.ndarray, theiler: int = 0) -> np.ndarray:
    """
    For each row i in X (shape mxd), find index of nearest neighbor under L_inf norm,
    excluding |i-j| <= theiler (Theiler window) and j=i.
    Returns nn_idx with shape (m,).
    """
    m = X.shape[0]
    if m <= 1:
        return np.zeros(m, dtype=int)

    # pairwise max-norm distances via broadcasting: O(m^2 * d)
    # dist[i, j] = max(|X[i] - X[j]|)
    # To save memory for large m, you could chunk this; here we keep it simple.
    diffs = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)  # (m, m)

    # mask self and Theiler window
    if theiler > 0:
        for i in range(m):
            lo = max(0, i - theiler)
            hi = min(m, i + theiler + 1)
            diffs[i, lo:hi] = np.inf
    np.fill_diagonal(diffs, np.inf)

    nn_idx = np.argmin(diffs, axis=1)
    return nn_idx

def cao_e1_e2(
    x: Sequence[float],
    tau: int = 1,
    d_max: int = 10,
    theiler: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute E1(d) and E2(d) for d=1..d_max (E1(d_max) is NaN).
    - Nearest neighbors under L_inf (max) norm
    - Theiler window excludes temporal neighbors (|i-j|<=theiler)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < (d_max + 1) * tau + 1:
        # Still proceed; higher d will become NaN as needed
        pass

    E1 = np.full(d_max, np.nan, dtype=float)
    E2 = np.full(d_max, np.nan, dtype=float)

    # Pre-embed once per d
    embeds = [None] * (d_max + 1)
    for d in range(1, d_max + 1):
        embeds[d] = _embed_trajectory(x, d, tau)

    for d in range(1, d_max + 1):
        Xd = embeds[d]
        m = Xd.shape[0]
        if m <= 1:
            continue

        nn_idx = _nearest_neighbors_maxnorm(Xd, theiler=theiler)

        # E1(d): ratio of distances in d+1 and d (only defined if d < d_max)
        if d < d_max:
            Xd1 = embeds[d + 1]
            m_next = min(Xd1.shape[0], m)  # valid points for both dims
            if m_next > 1:
                idx = np.arange(m_next)
                nn_cut = nn_idx[:m_next]
                # max-norm distances in d and d+1
                dist_d = np.max(np.abs(Xd[idx] - Xd[nn_cut]), axis=1)
                dist_d1 = np.max(np.abs(Xd1[idx] - Xd1[nn_cut]), axis=1)
                dist_d = np.where(dist_d == 0, np.finfo(float).eps, dist_d)
                E1[d - 1] = np.mean(dist_d1 / dist_d)

        # E2(d): mean |x_{i+d tau} - x_{n(i,d)+d tau}| / mean |x_i - x_{n(i,d)}|
        tail = d * tau
        if n - tail >= m and m > 1:
            num = np.abs(x[tail : tail + m] - x[nn_idx + tail])
            den = np.abs(x[:m] - x[nn_idx])
            den = np.where(den == 0, np.finfo(float).eps, den)
            E2[d - 1] = np.mean(num) / np.mean(den)

    return E1, E2

def estimate_embedding_dimension(E1: Sequence[float], tol: float = 0.01, start: int = 2) -> Optional[int]:
    """
    Return smallest (d+1) where E1 stabilizes: |E1(d+1) - E1(d)| < tol.
    If not found, return None.
    """
    E1 = np.asarray(E1, dtype=float)
    for d in range(max(1, start - 1), len(E1) - 1):
        if np.isfinite(E1[d]) and np.isfinite(E1[d + 1]):
            if abs(E1[d + 1] - E1[d]) < tol:
                return d + 2
    return None

# -------------------------
# IO helpers
# -------------------------
# 置き換え
def load_series_from_csv(path: str, col: Optional[str], sep: Optional[str]) -> np.ndarray:
    import pandas as pd, numpy as np
    if sep is None:
        # カンマ or 空白の自動判定
        try:
            df = pd.read_csv(path)
            if df.shape[1] == 1:
                # 1列しか無い＝空白区切りの可能性あり
                df2 = pd.read_csv(path, delim_whitespace=True)
                if df2.shape[1] > 1:
                    df = df2
        except Exception:
            # 空白区切りで再トライ
            df = pd.read_csv(path, delim_whitespace=True)
    else:
        if sep.strip() == "":
            df = pd.read_csv(path, delim_whitespace=True)
        else:
            df = pd.read_csv(path, sep=sep)

    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {path}")

    if col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        s = df[numeric_cols[0]].to_numpy() if numeric_cols else df.iloc[:, 0].to_numpy()
    else:
        if col.isdigit():
            s = df.iloc[:, int(col)].to_numpy()
        else:
            if col not in df.columns:
                raise ValueError(f'Column "{col}" not found in {path}. Available: {list(df.columns)}')
            s = df[col].to_numpy()

    s = np.asarray(s, dtype=float).reshape(-1)
    s = s[~np.isnan(s)]
    if s.size < 3:
        raise ValueError(f"Series in {path} too short after NaN removal.")
    return s

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="L. Cao method (E1, E2) for minimum embedding dimension.")
    ap.add_argument("--csv_files", nargs="+", required=True, help="CSV file(s) to process")
    ap.add_argument("--column", default=None, help="Column name or 0-based index (default: first numeric col)")
    ap.add_argument("--tau", type=int, default=1, help="Time delay τ (default: 1)")
    ap.add_argument("--d_max", type=int, default=12, help="Max embedding dimension d (default: 12)")
    ap.add_argument("--theiler", type=int, default=0, help="Theiler window size (exclude |i-j|<=theiler)")
    ap.add_argument("--normalize", choices=["zscore", "minmax"], default=None, help="Optional normalization")
    ap.add_argument("--tol", type=float, default=0.01, help="Stabilization tolerance for E1 (default: 0.01)")
    ap.add_argument("--save_prefix", default=None, help="If set, save E1/E2 table as CSV and plot as PNG per file")
    ap.add_argument("--no_stdout", action="store_true", help="Suppress printing arrays to stdout")
    args = ap.parse_args()

    # Lazy import matplotlib only if saving plot
    use_plot = args.save_prefix is not None
    if use_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[WARN] matplotlib not available; plots will be skipped.", file=sys.stderr)
            use_plot = False

    for path in args.csv_files:
        try:
            x = load_series_from_csv(path, args.column)
            x = maybe_normalize(x, args.normalize)
            E1, E2 = cao_e1_e2(x, tau=args.tau, d_max=args.d_max, theiler=args.theiler)
            m = estimate_embedding_dimension(E1, tol=args.tol, start=2)

            if not args.no_stdout:
                print(f"\n=== {path} ===")
                print(f"length={len(x)}, tau={args.tau}, d_max={args.d_max}, theiler={args.theiler}, normalize={args.normalize}")
                print("E1(d):", np.array2string(E1, precision=6, separator=", "))
                print("E2(d):", np.array2string(E2, precision=6, separator=", "))
                print("Estimated minimum embedding dimension (E1 stabilization):", m)

            if args.save_prefix:
                stem = f"{args.save_prefix}_{_safe_stem(path)}"
                # Save table
                out_df = pd.DataFrame({
                    "d": np.arange(1, args.d_max + 1, dtype=int),
                    "E1": E1,
                    "E2": E2
                })
                csv_out = f"{stem}_cao.csv"
                out_df.to_csv(csv_out, index=False)
                print(f"[Saved] {csv_out}")

                # Save plot if matplotlib available
                if use_plot:
                    import matplotlib.pyplot as plt  # type: ignore
                    # E1
                    plt.figure()
                    plt.plot(np.arange(1, args.d_max + 1), E1, marker="o")
                    plt.xlabel("d")
                    plt.ylabel("E1(d)")
                    plt.title(f"Cao E1(d): {path}")
                    plt.grid(True, alpha=0.3)
                    png1 = f"{stem}_E1.png"
                    plt.savefig(png1, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"[Saved] {png1}")
                    # E2
                    plt.figure()
                    plt.plot(np.arange(1, args.d_max + 1), E2, marker="o")
                    plt.xlabel("d")
                    plt.ylabel("E2(d)")
                    plt.title(f"Cao E2(d): {path}")
                    plt.grid(True, alpha=0.3)
                    png2 = f"{stem}_E2.png"
                    plt.savefig(png2, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"[Saved] {png2}")

        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)

def _safe_stem(path: str) -> str:
    import os
    base = os.path.basename(path)
    return os.path.splitext(base)[0].replace(" ", "_")

if __name__ == "__main__":
    main()
