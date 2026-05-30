#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E2-only (Ito method) comparator
- Always analyzes Lorenz (tau=5 fixed) and outputs E2 plot
- Optionally analyzes user real data (Method 1) and compares on the same figure
- Uniform noise is zero-mean; Gaussian noise mean=0
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


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
    """Return first lag where autocorr < 1/e, else max_lag."""
    ac = calculate_autocorrelation(series, max_lag)
    thr = 1 / np.e
    for i, v in enumerate(ac):
        if v < thr:
            return i + 1
    return max_lag


# ---------- embedding ----------
def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if s is None: s = tau
    if k is None: k = max(1, int(0.05 * N))

    E1 = []
    for m in range(1, max_dim + 1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0:
            break

        valid_M = M - s
        X_now = Xm[:valid_M]

        k_eff = min(k, valid_M - 1)
        if k_eff < 1:
            break
        over_k = min(valid_M - 1, k_eff + 20)

        nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm='auto', metric="euclidean")
        nn.fit(X_now)
        dists, indxs = nn.kneighbors(X_now)  # include itself

        ratios = []
        for i in range(valid_M):
            cand_j = indxs[i, 1:]  # 自身以外
            cand_d = dists[i, 1:]

            # Theiler window
            if theiler > 0:
                mask = np.abs(cand_j - i) > theiler
                cand_j = cand_j[mask]
                cand_d = cand_d[mask]

            if cand_j.size == 0:
                continue

            # 未来が有効 (i+s, j+s < M)
            mask2 = (cand_j + s) < M
            cand_j = cand_j[mask2]
            cand_d = cand_d[mask2]
            if cand_j.size == 0:
                continue

            # 上位k
            use = min(k_eff, cand_j.size)
            jj = cand_j[:use]
            d_now_i = cand_d[:use].mean()
            if d_now_i < 1e-12:
                continue

            # s ステップ後の距離
            d_fut_i = np.linalg.norm(Xm[i+s] - Xm[jj+s], axis=1).mean()
            ratios.append(d_fut_i / d_now_i)

        if not ratios:
            continue
        E1.append(float(np.mean(ratios)))

    return np.array(E1, dtype=float)

def estimate_min_dimension(E1, eps=0.05, win=3):
    E1 = np.asarray(E1, dtype=float)
    finite_mask = np.isfinite(E1)
    E1f = E1[finite_mask]
    if len(E1f) == 0:
        return 1
    if len(E1f) <= 1:
        return len(E1f)
    for m in range(1, len(E1) - win + 1):
        seg = E1[m:m+win]
        dif = np.abs(np.diff(seg))
        base = np.maximum(np.abs(seg[:-1]), 1e-12)
        if np.all(dif / base <= eps):
            return m + 1
    return len(E1)


# ---------- Ito E2 ----------
def _e_star(x, m, tau, k=None, theiler: int = 0):
    """
    E*(m): in m-dim space, use k nearest neighbors to average |(m+1)-th component differences|.
    """
    x = np.asarray(x, dtype=float)

    # m-dim embedding
    Xm = _embed(x, m, tau)
    M1 = Xm.shape[0]
    if M1 <= 0:
        return np.nan

    # indices for (m+1)-th component
    idx_last = np.arange(M1) + m * tau
    if np.any(idx_last >= len(x)):
        # truncate to valid indices
        ok = np.where(idx_last < len(x))[0]
        if ok.size == 0:
            return np.nan
        Xm = Xm[: ok[-1] + 1]
        idx_last = idx_last[: ok[-1] + 1]
        M1 = Xm.shape[0]
        if M1 <= 1:
            return np.nan

    # neighbor search
    k_default = max(1, int(0.05 * M1))  # 5% fallback
    k_eff = max(1, min(k if k is not None else k_default, M1 - 1))
    over_k = min(M1 - 1, k_eff + 20)

    nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm="auto", metric="euclidean")
    nn.fit(Xm)
    dists, index = nn.kneighbors(Xm)  # includes self at [:,0]

    diffs = []
    for i in range(M1):
        cand = index[i, 1:]  # drop self
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
    """
    Compute E2(m) = E*(m+1) / E*(m)  for m = 1..max_dim-1
    """
    x = np.asarray(x, dtype=float)

    # E*(m)
    Estar = np.array([_e_star(x, m, tau, k, theiler) for m in range(1, max_dim + 1)], dtype=float)

    # ratio
    E2 = []
    for m in range(1, max_dim):
        a, b = Estar[m - 1], Estar[m]
        if not np.isfinite(a) or a == 0 or not np.isfinite(b):
            E2.append(np.nan)
        else:
            E2.append(b / a)
    return np.array(E2, dtype=float)  # length = max_dim-1

    # ----- Method 1: real data -----
def run_method1_cao(series, max_dim=10, k_frac=0.05, theiler=0):
    series = np.asarray(series, dtype=float)
    tau = determine_tau(series)
    k = int(max(1, k_frac * len(series)))
    e1_vals = itho_e1(series, max_dim=max_dim, tau=tau, s=tau, k=k, theiler=theiler)
    m_axis = list(range(1, len(e1_vals) + 1))
    return m_axis, e1_vals, tau

# ----- Method 2: lorenz.csv -----
def run_method2_cao(series, max_dim=10, k_frac=0.05, theiler=0, add_noise=False, rng=None):
    x = np.asarray(series, dtype=float)
    tau = 5
    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        # 一様ノイズを加算
        x = x + rng.uniform(0.0, 1.0, size=x.shape)
    k = int(max(1, k_frac * len(x)))
    e1_vals = itho_e1(x, max_dim=max_dim, tau=tau, s=tau, k=k, theiler=theiler)
    m_axis = list(range(1, len(e1_vals) + 1))
    return m_axis, e1_vals

# ---------- I/O / Plot ----------
def load_series_whitespace_firstcol(path: str):
    # future-proof: avoid delim_whitespace deprecation
    df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    return df.iloc[:, 0].values

def plot_e2_curves(curves, savepath, max_dim=10, title="Ito E2 vs Embedding Dimension"):
    plt.figure(figsize=(10, 6))
    # baseline guide
    plt.hlines(y=1.0, xmin=1, xmax=max_dim - 1, linestyles="dashed", colors="gray",
               label="White Noise Reference (E2≈1)")
    for c in curves:
        m = c["m"]; y = c["E2"]
        plt.plot(m, y, marker=c.get("marker", "o"), linestyle=c.get("linestyle", "-"),
                 linewidth=1.6, label=c["label"])
    plt.xlabel("Embedding Dimension m")
    plt.ylabel("Ito E2 (= E*(m+1)/E*(m))")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(range(1, max_dim))
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"[Saved] {savepath}")


def compute_e2_for_series(x, max_dim, tau, k, theiler, label, marker):
    e2 = itho_e2(x, max_dim=max_dim, tau=tau, k=k, theiler=theiler)
    return {
        "m": list(range(1, len(e2) + 1)),  # m = 1..max_dim-1
        "E2": e2,
        "label": label,
        "marker": marker,
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Ito E2 comparison only (always includes Lorenz; tau=5 fixed for Lorenz)"
    )
    parser.add_argument(
        "--method1_csv", nargs="+", required=False,
        help="Real-data files (whitespace-delimited; first column is used). Optional."
    )
    parser.add_argument(
        "--lorenz_csv", default="lorenz.csv",
        help="Path to lorenz.csv (whitespace-delimited; first column). REQUIRED."
    )
    parser.add_argument("--output_dir", default="finalresult_ito_e2only",
                        help="Directory to save outputs")
    parser.add_argument("--max_dim", type=int, default=10)
    parser.add_argument("--k_nn", type=int, default=None,
                        help="k neighbors in neighbor search (default: 5% of samples)")
    parser.add_argument("--theiler", type=int, default=0)
    parser.add_argument("--noise_type", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, remove output_dir before writing.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # output dir
    if os.path.exists(args.output_dir):
        if args.overwrite:
            import shutil
            print(f"[INFO] Removing existing directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            print(f"[INFO] Using existing directory: {args.output_dir}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] Created: {args.output_dir}")

    e2_curves = []

    # ----- Method 1: Real data (optional). tau is estimated on standardized data for stability.
    if args.method1_csv:
        for f in args.method1_csv:
            base = os.path.basename(f)
            if not os.path.exists(f):
                print(f"[WARN] Missing: {f} (skip)")
                continue
            x_raw = load_series_whitespace_firstcol(f)
            # standardize before tau estimation (E2 is scale-invariant but tau is more stable)
            x_std = (x_raw - np.mean(x_raw)) / (np.std(x_raw) or 1.0)
            tau1 = determine_tau(x_std, max_lag=100)
            print(f"[INFO] tau (auto) for {base}: {tau1}")

            e2_curves.append(
                compute_e2_for_series(
                    x_raw, max_dim=args.max_dim, tau=tau1,
                    k=args.k_nn, theiler=args.theiler,
                    label=f"Method 1 ({base})", marker="o"
                )
            )

    # ----- Method 2: Lorenz (ALWAYS; tau=5 fixed)
    if not os.path.exists(args.lorenz_csv):
        raise FileNotFoundError(f"{args.lorenz_csv} not found (Lorenz is required).")
    lorenz = load_series_whitespace_firstcol(args.lorenz_csv)
    print("[INFO] Lorenz: tau fixed = 5")

    # (a) no noise
    e2_curves.append(
        compute_e2_for_series(
            lorenz, max_dim=args.max_dim, tau=5,
            k=args.k_nn, theiler=args.theiler,
            label="Method 2 (lorenz, no noise)", marker="s"
        )
    )

    # (b) with noise (zero-mean uniform or gaussian)
    if args.noise_type == "uniform":
        noise = np.random.uniform(-args.noise_scale, args.noise_scale, size=lorenz.shape)
    else:  # gaussian
        noise = np.random.normal(0, args.noise_scale, size=lorenz.shape)

    e2_curves.append(
        compute_e2_for_series(
            lorenz + noise, max_dim=args.max_dim, tau=5,
            k=args.k_nn, theiler=args.theiler,
            label=f"Method 2 (lorenz + {args.noise_type})", marker="^"
        )
    )

    # ----- Save Plot -----
    savepath_e2 = os.path.join(args.output_dir, "comparison_e2.png")
    plot_e2_curves(e2_curves, savepath_e2, max_dim=args.max_dim,
                   title="Ito E2 vs Embedding Dimension (Method1 / Method2)")

    print("\n[Done] E2-only comparison finished.")


if __name__ == "__main__":
    main()
