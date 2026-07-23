#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Itho_middle_Each.py
# Ito's E1 指標で Method1(実データ) / Method2(lorenz.csv, ノイズ有無) / 白色ノイズ基準 を1枚の図で比較

#指定したデータを解析することが出来る
#トランスレーション誤差との比較用の解析結果を出力するために作成

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl

mpl.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 20,   # X軸ラベル・Y軸ラベル
    "axes.titlesize": 22,   # タイトル
})


# --------- 伊藤法（E1）の実装 --------- #
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
    x = np.asarray(x, dtype=float)
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if s is None:
        s = tau
    if k is None:
        k = max(1, int(0.05 * N))  # デフォルト: 長さの5%

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
        dists, indxs = nn.kneighbors(X_now)  # 自身を含む

        ratios = []
        for i in range(valid_M):
            cand_j = indxs[i, 1:]   # 自身以外
            cand_d = dists[i, 1:]

            # Theiler window
            if theiler > 0:
                mask = np.abs(cand_j - i) > theiler
                cand_j = cand_j[mask]
                cand_d = cand_d[mask]
            if cand_j.size == 0:
                continue

            # 未来が有効
            mask2 = (cand_j + s) < M
            cand_j = cand_j[mask2]
            cand_d = cand_d[mask2]
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

# --------- ラッパー --------- #
def run_method1_ito(series, max_dim=10, s=None, k=None, theiler=0, auto_tau=True):
    x = np.asarray(series, dtype=float)
    # 標準化（Method1は前スクリプトと同様に正規化）
    mu, sd = np.mean(x), np.std(x)
    x = (x - mu) / (sd if sd > 0 else 1.0)

    tau = determine_tau(x, max_lag=100) if auto_tau else 5
    E1 = itho_e1(x, max_dim=max_dim, tau=tau, s=s, k=k, theiler=theiler)
    return list(range(1, len(E1) + 1)), E1, tau  # τ は内部利用のみ

def run_method2_ito(series, max_dim=10, tau=5, s=None, k=None, theiler=0,
                    add_noise=False, noise_type="uniform", noise_scale=1.0):
    x = np.asarray(series, dtype=float)
    if add_noise:
        if noise_type == "uniform":
            noise = np.random.uniform(0, noise_scale, size=x.shape)
        else:
            noise = np.random.normal(0, noise_scale, size=x.shape)
        x = x + noise
    # Method2 は前スクリプト同様、標準化はしない
    E1 = itho_e1(x, max_dim=max_dim, tau=tau, s=s, k=k, theiler=theiler)
    return list(range(1, len(E1) + 1)), E1

# --------- I/O / Plot --------- #
def load_series_whitespace_firstcol(path):
    df = pd.read_csv(path, header=None, delim_whitespace=True)
    return df.iloc[:, 0].values

def plot_e1_curves(curves, savepath, max_dim=10, title="Ito E1 vs Embedding Dimension",
                   mstar_report=None):

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hlines(y=1.0, xmin=1, xmax=max_dim,
              colors='gray', linestyles='dashed',
              label='White Noise Reference (E1≈1)')

    for c in curves:
        m = c['m']; y = c['E1']
        ax.plot(m, y, marker=c.get('marker', 'o'), linestyle=c.get('linestyle', '-'),
                linewidth=1.6, label=c['label'])

    ax.set_xlabel('Embedding Dimension m')
    ax.set_ylabel('Ito E1 (⟨d_future/d_now⟩)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.set_xticks(range(1, max_dim + 1))

    # ★ここがポイント：座標系を ax.transAxes に固定
    leg = ax.legend(loc='upper left',
                    bbox_to_anchor=(0, 1),
                    bbox_transform=ax.transAxes,
                    frameon=True,
                    borderaxespad=0.0)

    # 左上テキスト
    if mstar_report:
        lines = [f"{name}: m*={mstar}" for name, mstar in mstar_report]
        text_str = "\n".join(lines)
        ax.text(0.01, 0.99, text_str,
                transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    fig.savefig(savepath, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# --------- Main --------- #
def main():
    parser = argparse.ArgumentParser(description="Ito E1 comparison (Method1/Method2/Baseline)")
    parser.add_argument("--method1_csv", nargs="+", required=True,
                        help="Whitespace-delimited files for Method 1 (first column used)")
    parser.add_argument("--lorenz_csv", default="lorenz.csv",
                        help="Path to lorenz.csv (whitespace-delimited, first column). If missing, Method 2 is skipped.")
    parser.add_argument("--output_dir", default="finalresult_ito",
                        help="Directory to save outputs")
    parser.add_argument("--max_dim", type=int, default=10)
    parser.add_argument("--tau2", type=int, default=5, help="Fixed tau for Method 2")
    parser.add_argument("--lead_s", type=int, default=None, help="Lead s (default: tau)")
    parser.add_argument("--k_nn", type=int, default=None, help="k neighbors (default: 5% of length)")
    parser.add_argument("--theiler", type=int, default=0)
    parser.add_argument("--noise_type", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # m* 推定用パラメータ
    parser.add_argument("--mstar_eps", type=float, default=0.15,
                      help="Stabilization threshold on smoothed d(log E1)/dm (recommend 0.12–0.2)")
    parser.add_argument("--mstar_consec", type=int, default=2,
                        help="Consecutive steps to accept stabilization")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 出力ディレクトリ
    if os.path.exists(args.output_dir):
        print(f"[INFO] Removing existing directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Created: {args.output_dir}")

    curves = []
    mstar_report = []  # (name, m_star)

    
    # ----- Method 1（実データ） -----
    for f in args.method1_csv:
        base = os.path.basename(f)
        if not os.path.exists(f):
            print(f"[ERROR] Cannot find {f}")
            continue
        x = load_series_whitespace_firstcol(f)
        m1, e1, tau1 = run_method1_ito(
            x, max_dim=args.max_dim,
            s=args.lead_s, k=args.k_nn, theiler=args.theiler, auto_tau=True
        )
        print(f"[INFO] tau (delay time) for {base}: {tau1}")
        # estimate_min_dimension は (eps, win) シグネチャ
        mstar_1 = estimate_min_dimension(e1, eps=args.mstar_eps, win=args.mstar_consec)
        mstar_report.append((f"Method 1 ({base})", mstar_1))
        curves.append({
            'm': m1, 'E1': e1,
            'label': f"Method 1 ({base}), m*={mstar_1}",
            'marker': 'o'
        })


    # ----- Method 2（lorenz.csv があれば実施） -----
    if os.path.exists(args.lorenz_csv):
        lorenz = load_series_whitespace_firstcol(args.lorenz_csv)
        print(f"[INFO] tau (delay time) for Method 2: {args.tau2}")
        m2_a, e2_a = run_method2_ito(
            lorenz, max_dim=args.max_dim, tau=args.tau2,
            s=args.lead_s, k=args.k_nn, theiler=args.theiler,
            add_noise=False, noise_type=args.noise_type, noise_scale=args.noise_scale
        )
       
        mstar_a = estimate_min_dimension(e2_a, eps=args.mstar_eps, win=args.mstar_consec)
        name_a = f"Method 2 (lorenz, no noise)"
        mstar_report.append((name_a, mstar_a))
        curves.append({
            'm': m2_a, 'E1': e2_a,
            'label': f"{name_a}, m*={mstar_a}",  # ← m=1..{args.max_dim} を削除
            'marker': 's'
        })
        m2_b, e2_b = run_method2_ito(
            lorenz, max_dim=args.max_dim, tau=args.tau2,
            s=args.lead_s, k=args.k_nn, theiler=args.theiler,
            add_noise=True, noise_type=args.noise_type, noise_scale=args.noise_scale
        )
        mstar_b = estimate_min_dimension(e2_b, eps=args.mstar_eps, win=args.mstar_consec)
        name_b = f"Method 2 (lorenz + {args.noise_type})"
        mstar_report.append((name_b, mstar_b))
        curves.append({
            'm': m2_b, 'E1': e2_b,
            'label': f"{name_b}, m*={mstar_b}",
            'marker': '^'
        })
    else:
        print(f"[WARN] {args.lorenz_csv} not found. Skipping Method 2.")

    # ----- 図の保存 -----
    savepath = os.path.join(args.output_dir, "comparison_e1.png")
    plot_e1_curves(curves, savepath, max_dim=args.max_dim,
                   title="Ito E1 vs Embedding Dimension (Method1 / Method2 / Baseline)")

    # ----- m* 一覧を表示 -----
    if mstar_report:
        print("\n[INFO] Estimated minimal embedding dimension (m*):")
        for name, mstar in mstar_report:
            print(f"  - {name}: m* = {mstar}")

    print("\n[Done] E1 comparison finished.")

if __name__ == "__main__":
    main()
