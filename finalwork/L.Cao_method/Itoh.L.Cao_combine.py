#2025/09/09
#実際のデータとlorenzのデータを比較可能なプログラム
#Method 1が実際のデータを解析
#Method 2では、Lorenzアトラクタのノイズがあるものとないものを解析比較

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import glob
import argparse
import shutil
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def calculate_autocorrelation(series, max_lag):
    autocorr_values = []
    series = np.asarray(series, dtype=float)
    series_mean = np.mean(series)
    denominator = np.sum((series - series_mean) ** 2)
    for lag in range(1, max_lag + 1):
        if lag >= len(series):
            break
        num = np.sum((series[:-lag] - series_mean) * (series[lag:] - series_mean))
        autocorr = num / denominator if denominator != 0 else 0.0
        autocorr_values.append (autocorr)
    return autocorr_values

def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threshold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1
    return max_lag

def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim = 10, tau = 5, s = None, k = None, theiler = 0):
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
        dists, indxs = nn.kneighbors(X_now) #include itself

        ratios = []
        for i in range(valid_M):
            cand_j = indxs[i, 1:]            #自身以外
            cand_d = dists[i, 1:]

            #Theiler windowで除外
            if theiler > 0:
                mask = np.abs(cand_j - i) > theiler
                cand_j = cand_j[mask]
                cand_d = cand_d[mask]
            
            if cand_j.size == 0:
                continue

            #未来が有効な近傍だけ(i+s, j+s < M)
            mask2 = (cand_j + s) < M
            cand_j = cand_j[mask2]
            cand_d = cand_d[mask2]
            if cand_j.size == 0:
                continue

            #上位k を使用
            use = min(k_eff, cand_j.size)
            jj = cand_j[:use]
            d_now_i = cand_d[:use].mean()
            if d_now_i < 1e-12:
                continue

            #s ステップ後の距離
            d_fut_i = np.linalg.norm(Xm[i+s] - Xm[jj+s], axis=1).mean()
            ratios.append(d_fut_i / d_now_i)


        if not(ratios):
            continue
        E1.append(float(np.mean(ratios)))
    
    return np.array(E1, dtype = float)

def estimate_min_dimension(E1, eps=0.05, win = 3):
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

#-----Method 1:real data-----
def run_method1_cao(series, max_dim = 10, k_frac = 0.05,  theiler = 0):
    print(f"\n===== [ Method 1 ] Start =====")
    series = np.asarray(series, dtype = float)
    tau = determine_tau(series)
    print(f"[INFO] τ (auto) = {tau}")

    k = int(max(1, k_frac * len(series)))
    e1_vals = itho_e1(series,  max_dim=max_dim, tau=tau, s=tau, k=k, theiler=theiler)
    m_axis = list(range(1, len(e1_vals)+1))
    return m_axis, e1_vals, tau

#-----Method 2: lorenz.csv-----
def run_method2_cao(series, max_dim = 10, k_frac = 0.05,  theiler = 0, add_noise=False, rng=None):
    x = np.asarray(series, dtype = float)
    tau = 5
    if add_noise:
        print("[INFO] Adding uniform noise to series in Method 2")
        if rng is None:
            rng = np.random.default_rng()
        x = x + rng.uniform(0.0, 1.0, size=x.shape)
    k = int(max(1, k_frac * len(x)))
    e1_vals = itho_e1(x,max_dim=max_dim, tau=tau, s=tau, k=k, theiler=theiler)
    m_axis = list(range(1, len(e1_vals)+1))
    return m_axis, e1_vals

#-----Main-----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="Itho.Cao_result",
                        help="Directory to save plot")
    args = parser.parse_args()
    
    #固定パラメータ設定
    max_dim = 10
    k_frac = 0.50
    theiler = 0
    eps = 0.05
    win = 3
    seed = 42


    #output dir
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Created: {args.output_dir}")

    #RNG
    rng = np.random.default_rng(seed) 

    #-----Method 2-----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lorenz_file = os.path.join(script_dir, "lorenz.csv")

    print("\n=== Computing Method 2 using fixed lorenz.csv ===")
    if not os.path.exists(lorenz_file):
        print(f"[ERROR] Cannot find {lorenz_file}")
        return

    series2 = pd.read_csv(lorenz_file, header=None, delim_whitespace=True).iloc[:, 0].values
    m2_no_noise, itho_e2_no_noise = run_method2_cao(series2, max_dim=max_dim,
                                                k_frac=k_frac, theiler=theiler,
                                                add_noise=False, rng=rng)
    m2_with_noise, itho_e2_with_noise = run_method2_cao(series2, max_dim=max_dim,
                                                k_frac=k_frac, theiler=theiler,
                                                add_noise=False, rng=rng)
    
    #------Method 1 (lc/*.csv)-----
    plt.figure(figsize=(10,6))
    method1_files = sorted(glob.glob(os.path.join("lc", "*.csv")))
    if not method1_files:
        print("[INFO] No CSV files gound under 'lc/' directory")
    
    for method1_file in method1_files:
        print(f"\n====Processing {method1_file} for Method 1=====")
        series1 = pd.read_csv(method1_file, header=None, delim_whitespace=True).iloc[:, 0].values
        m1, itho_e1, tau1 = run_method1_cao(series1, max_dim=max_dim,
                                            k_frac=k_frac, theiler=theiler)
        plt.plot(m1, itho_e1, marker='o',
                 label=f"Method 1({os.path.basename(method1_file)}, tau={tau1})")
    
    #White noise 基準線
    plt.hlines(y=1.0, xmin=1, xmax=max_dim,
               colors='gray', linestyles='dashed',
               label='White Noise Reference (E1≈1)')
    
    #Method 2 ライン 追加
    plt.plot(m2_no_noise, itho_e2_no_noise, marker='s', label="Method 2 (lorenz, No Noise, tau=5)")
    plt.plot(m2_with_noise, itho_e2_with_noise, marker='^', label="Method 2 (lorenz, With Noise, tau=5)")

    plt.xlabel('Embedding Dimension m')
    plt.ylabel("Cao's E1")
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', linewidth=0.5)
    plt.xticks(range(1, max_dim + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    savepath = os.path.join(args.output_dir, "cao_comparison_plot.png")
    plt.savefig(savepath, dpi=200)
    plt.close()
    print(f"[Saved] {savepath}")
    print("\n[Done] Cao (E1) comparison finished.")

if __name__ == "__main__":
    main()
