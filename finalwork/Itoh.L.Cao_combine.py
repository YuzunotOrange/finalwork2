# 2025/09/09
# 各 lc/*.csv の解析結果と lorenz.csv（ノイズなし/あり）を
# 1ファイルずつ比較グラフとして出力する版

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
import re


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
        autocorr_values.append(autocorr)
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

def sanitize_filename(name: str) -> str:
    # 拡張子を除いたベース名を安全なファイル名に
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^\w\-.]+", "_", base)
    return base or "output"

# ----- Main -----
def main():

    result_summary = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="Itho.Cao_result",
                        help="Directory to save plots")
    parser.add_argument("--max_dim", type=int, default=10)
    parser.add_argument("--k_frac", type=float, default=0.50)
    parser.add_argument("--theiler", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overall_plot", action="store_true",
                        help="全ファイル重ね合わせの総合図も出力する")
    args = parser.parse_args()

    max_dim = args.max_dim
    k_frac = args.k_frac
    theiler = args.theiler
    seed = args.seed

    # 出力ディレクトリ
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Created: {args.output_dir}")

    rng = np.random.default_rng(seed)

    result_summary = []

    # ----- Lorenz を一度だけ読み計算 -----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lorenz_file = os.path.join(script_dir, "lorenz.csv")
    if not os.path.exists(lorenz_file):
        print(f"[ERROR] Cannot find {lorenz_file}")
        return
    


    # 空白区切り対応（delim_whitespace の将来警告回避）
    series2 = pd.read_csv(lorenz_file, header=None, sep=r"\s+").iloc[:, 0].values

    print("\n=== Computing Method 2 (Lorenz) ===")
    m2_no_noise, e2_no_noise = run_method2_cao(
        series2, max_dim=max_dim, k_frac=k_frac, theiler=theiler, add_noise=False, rng=rng
    )
    m2_with_noise, e2_with_noise = run_method2_cao(
        series2, max_dim=max_dim, k_frac=k_frac, theiler=theiler, add_noise=True, rng=rng
    )

    # ----- new_li/*.csv ごとに個別図を作成 -----
    method1_files = sorted(glob.glob(os.path.join("new_li", "*.csv")))
    if not method1_files:
        print("[INFO] No CSV files found under 'new_li/' directory")
    else:
        print(f"[INFO] Found {len(method1_files)} files under new_li/")

    # 総合図用（任意）
    if args.overall_plot:
        plt.figure(figsize=(10, 6))

    for method1_file in method1_files:
        print(f"\n==== Processing {method1_file} ====")
        # lc 側はカンマ区切り想定
        series1 = pd.read_csv(method1_file, header=None, sep=",").iloc[:, 0].values
        m1, e1_vals, tau1 = run_method1_cao(
            series1, max_dim=max_dim, k_frac=k_frac, theiler=theiler
        )

        if len(e1_vals) > 0:
            min_dim = estimate_min_dimension(e1_vals)
            print(f"Estimated minimum dimension for {os.path.basename(method1_file)}: {min_dim}")
        else:
            min_dim = None

        #summaryを追加
        result_summary.append({
            "filename": os.path.basename(method1_file),
            "tau": tau1,
            "min_dim":min_dim
        })

        # --- 個別図 ---
        plt.figure(figsize=(10, 6))

        # White noise 参考線
        plt.hlines(y=1.0, xmin=1, xmax=max_dim,
                   colors='gray', linestyles='dashed',
                   label='White Noise Reference (E1≈1)')

        # Method 1（そのファイル）
        if len(e1_vals) > 0:
            label_name = f"Method 1 ({os.path.basename(method1_file)}, tau={tau1}"
            if min_dim is not None:
                label_name += f", min_dim={min_dim}"
        label_name += ")"

        plt.plot(m1, e1_vals, marker='o', label=label_name)

        # Method 2（Lorenz: No Noise / With Noise）
        plt.plot(m2_no_noise, e2_no_noise, marker='s',
                 label="Method 2 (Lorenz, No Noise, tau=5)")
        plt.plot(m2_with_noise, e2_with_noise, marker='^',
                 label="Method 2 (Lorenz, With Noise, tau=5)")

        plt.xlabel('Embedding Dimension m')
        plt.ylabel("Cao\'s E1")
        plt.yscale('log')
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.xticks(range(1, max_dim + 1), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='upper right', framealpha=0.8, facecolor="white")
        plt.tight_layout()

        base = sanitize_filename(method1_file)
        savepath = os.path.join(args.output_dir, f"cao_compare_{base}.png")
        plt.savefig(savepath, dpi=200)
        plt.close()
        print(f"[Saved] {savepath}")

        # 総合図にも Method1 を追加（任意）
        if args.overall_plot and len(e1_vals) > 0:
            plt.plot(m1, e1_vals, marker='o', label=f"{os.path.basename(method1_file)} (tau={tau1})")

    # 総合図の保存（任意）
    if args.overall_plot and method1_files:
        # Lorenz 系も重ねる
        plt.plot(m2_no_noise, e2_no_noise, marker='s', label="Lorenz No Noise (tau=5)")
        plt.plot(m2_with_noise, e2_with_noise, marker='^', label="Lorenz With Noise (tau=5)")
        plt.hlines(y=1.0, xmin=1, xmax=max_dim, colors='gray', linestyles='dashed', label='E1≈1')

        plt.xlabel('Embedding Dimension m')
        plt.ylabel("Cao\'s E1")
        plt.yscale('log')
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.xticks(range(1, max_dim + 1), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.tight_layout()

        savepath_all = os.path.join(args.output_dir, "cao_comparison_all_files.png")
        plt.savefig(savepath_all, dpi=200)
        plt.close()
        print(f"[Saved] {savepath_all}")

    #summary csvを保存する
    if result_summary:
        df_summary = pd.DataFrame(result_summary)
        csv_path = os.path.join(args.output_dir, "result_summary.csv")
        df_summary.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[Saved] Summary CSV: {csv_path}")
        
        md_path = os.path.join(args.output_dir, "result_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df_summary.to_markdown(index=False))
        print(f"[Saved] Summary Markdown: {md_path}")

    print("\n[Done] Per-file Cao (E1) comparison finished.")

    

    
if __name__ == "__main__":
    main()
