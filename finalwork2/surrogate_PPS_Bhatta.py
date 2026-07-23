# 2026/06/04
#線形補間の影響が非線形性に対してどれだけ影響を及ぼすのか確認できる
#サロゲートデータ法の実装に関するプログラム
#生成法は偽ピリオディックサロゲート
#統計量として佐野、沢田法(1987)に基づいてリアプノフ指数を使用している
#検定方法はモンテカルロ有意性検定
#遅れ時間は相互情報量、埋め込み次元は伊藤法E1で推定
#欠損の方法として欠損部位をブロックごとランダムに変更（pattern 1~3)
#Bhttaの20個のデータを処理

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import os
import sys
import datetime
import random
import shutil
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import tkinter as tk
from tkinter import filedialog


base_dir = os.path.dirname(os.path.abspath(__file__))

result_dir = os.path.join(
    base_dir,
    "result_PPS_Bhatta"
)

if os.path.exists(result_dir):
    print(f"Cleaning up {result_dir}...")
    shutil.rmtree(result_dir)

os.makedirs(result_dir, exist_ok=True)

print(f"Result directory: {result_dir}")


def load_csv_timeseries(csv_path):

    df = pd.read_csv(
        csv_path,
        header=None
    )

    return df.iloc[:, 0].astype(float).values


# ====== 破損線形補間を施している ======
def introduce_block_missing_and_interpolate(x, missing_rate=0.1, block_len=10, seed=None, pattern = "random"):
    """
    連続ブロック欠損をランダムに付与し、線形補間で埋める。
    missing_rate : 全体の何割を欠損させるか (0 <= r < 1)
    block_len    : 1ブロックの欠損長（サンプル数）
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if missing_rate <= 0:
        return x.copy(), x.copy(), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)

    total_missing = int(round(missing_rate * n))
    num_blocks = max(1, total_missing // block_len)

    mask = np.zeros(n, dtype=bool)

    diff = np.abs(np.diff(x))

    if pattern == "random":
        candidates = np.arange(n - block_len)
    
    elif pattern == "high":
        
        #上位20%を高変化領域とする
        thresh = np.percentile(diff, 60)

        candidates = np.where(diff >= thresh)[0]
    
    elif pattern == "low":

        #下位20%を低変化領域とする
        thresh = np.percentile(diff, 40)

        candidates = np.where(diff <= thresh)[0]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    candidates = candidates[candidates < n - block_len]

    #ブロック配置
    placed = 0
    trial = 0
    max_trial = 20000

    while placed < num_blocks and trial < max_trial:
        
        start = rng.choice(candidates)

        if not mask[start:start + block_len].any():
            mask[start:start + block_len] = True
            placed += 1
        
        trial += 1
    
    if placed < num_blocks:

        remain = np.where(~mask)[0]

        extra = total_missing - mask.sum()

        if extra > 0:

            extra = min(
                extra,
                    len(remain)
            )

            add_idx = rng.choice(
                    remain,
                    size=extra,
                    replace=False
            )

            mask[add_idx] = True


    # 上限調整
    if mask.sum() > total_missing:
        extra = mask.sum() - total_missing
        idx_true = np.where(mask)[0]
        remove_idx = rng.choice(idx_true, size=extra, replace=False)
        mask[remove_idx] = False


    y_missing = x.copy()
    y_missing[mask] = np.nan

    # 線形補間（端点は最近傍値で補完）
    xi = np.arange(n)
    valid = ~np.isnan(y_missing)
    y_filled = np.interp(
        xi,
        xi[valid],
        y_missing[valid]
    )
    print(
    f"[{pattern}] "
    f"candidates={len(candidates)} "
    f"placed={placed}/{num_blocks} "
    f"actual_missing={mask.sum()/n:.3f}"
    )

    return y_missing, y_filled, mask

def average_mutual_information(x, max_lag=100, bins=32):

    x = np.asarray(x)

    ami = []

    for lag in range(1, max_lag + 1):

        x1 = x[:-lag]
        x2 = x[lag:]

        # ヒストグラム分割
        x1_bin = np.digitize(
            x1,
            np.histogram_bin_edges(x1, bins=bins)
        )

        x2_bin = np.digitize(
            x2,
            np.histogram_bin_edges(x2, bins=bins)
        )

        mi = mutual_info_score(x1_bin, x2_bin)

        ami.append(mi)

    return np.array(ami)

def determine_tau(series, max_lag=100):

    ami = average_mutual_information(
        series,
        max_lag=max_lag
    )

    # 最初の極小値
    for i in range(1, len(ami)-1):

        if ami[i] < ami[i-1] and ami[i] < ami[i+1]:
            return i + 1

    return np.argmin(ami) + 1

def _embed(x, m, tau):
    N = len(x) - (m - 1)*tau
    if N <= 0: return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau*np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    if s is None: s = tau
    if k is None: k = 10
    E1 = []
    for m in range(1, max_dim+1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0: break
        valid_M = M - s
        X_now = Xm[:valid_M]
        k_eff = min(k, valid_M-1)
        if k_eff < 1: break
        nn = NearestNeighbors(n_neighbors=min(k_eff+21, valid_M), algorithm='kd_tree').fit(X_now)
        dists, idxs = nn.kneighbors(X_now)
        ratios = []
        for i in range(valid_M):
            j = idxs[i, 1:]
            d = dists[i, 1:]
            if theiler > 0:
                mask = np.abs(j - i) > theiler
                j, d = j[mask], d[mask]
            if j.size == 0: continue
            mask2 = (j + s) < M
            j, d = j[mask2], d[mask2]
            if j.size == 0: continue
            use = min(k_eff, j.size)
            jj = j[:use]
            dn = np.mean(d[:use])
            if dn < 1e-12: continue
            df = np.mean(np.linalg.norm(Xm[i+s] - Xm[jj+s], axis=1))
            ratios.append(df/dn)
        if ratios: E1.append(np.mean(ratios))
    return np.array(E1)


# --- 佐野、沢田法（最大リアプノフ指数推定）---
def sano_sawada_lyapunov(data, m = 3, tau = 1, n_neighbors = 30, theiler=None):
    
    N = len(data)
    #アトラクタの再構築
    embedded = np.array([data[i:N - (m-1)*tau + i:tau] for i in range(m)]).T
    num_points = len(embedded)

    lyap_exp = []
    #接ベクトルの時間発展を計算
    step = 1 #写像の場合は1ステップづつ、連続系なら間隔を調整
    
    for i in range(0, num_points - step, 10):
        search_range = num_points - step
        diff = embedded[:search_range] - embedded[i]
        
        dist = np.linalg.norm(diff, axis=1)

        # 自分自身を除外
        dist[i] = np.inf

        # Theiler Window
        if theiler is None:
            theiler = tau * m
        w = theiler
        start = max(0, i - w)
        end = min(search_range, i + w + 1)

        dist[start:end] = np.inf

        #近傍点のインデックスを取得
        nearest_idx = np.argsort(dist)[:n_neighbors]

        #ヤコビ行列の推定用行列構成
        z = embedded[nearest_idx] - embedded[i]
        z_next = embedded[nearest_idx + step] -embedded[i + step]

        try:
            #最小二乗法で A (z_next = A * z)を推定
            A, _, _, _ = np.linalg.lstsq(z, z_next, rcond=None)
            #特異値分解から最大伸長率を抽出
            singular_values = np.linalg.svd(A, compute_uv=False)
            if singular_values[0] > 0:
                lyap_exp.append(np.log(singular_values[0]))
        except:
            continue
    return np.mean(lyap_exp) if lyap_exp else 0

def pps_surrogate(x, m, tau, r=None, return_indices=False):

    x = np.asarray(x)

    X = _embed(x, m, tau)

    M = len(X)

    if M < 20:

        if return_indices:
            return x.copy(), np.arange(len(x))
        
        return x.copy()

    if r is None:

        sample = X[
            np.random.choice(
                M,
                min(1000, M),
                replace=False
            )
        ]

        dist = np.linalg.norm(
            sample[:, None] - sample[None, :],
            axis=2
        )

        r = np.percentile(
            dist[dist > 0],
            5
        )

    idx = np.random.randint(M)

    surrogate_idx = [idx]

    for _ in range(M - 1):

        current = X[idx]

        dist = np.linalg.norm(
            X - current,
            axis=1
        )

        prob = np.exp(-dist / r)

        theiler = tau * m
        
        prob[np.abs(np.arange(M) - idx) <= theiler] = 0

        prob[idx] = 0

        prob_sum = prob.sum()

        if prob_sum == 0:
            j = np.random.randint(M)
        else:
            prob /= prob_sum
            j = np.random.choice(M, p=prob)

        idx = min(j + 1, M - 1)

        surrogate_idx.append(idx)

    surrogate_idx = np.array(surrogate_idx)

    if return_indices:
        return x[surrogate_idx], surrogate_idx

    return x[surrogate_idx]

def significance_label(z):
    if np.isnan(z):
        return "NnN"
    
    if abs(z) > 2.58:
        return "1%"
    
    elif abs(z) > 1.96:
        return "5%"
    
    else:
        return "NS"

def surrogate_z_test(original_value, surrogate_values):

    surrogate_values = np.asarray(surrogate_values)

    mu = np.mean(surrogate_values)
    sigma = np.std(surrogate_values, ddof=1)

    if sigma == 0:
        z = np.nan
    else:
        z = (original_value - mu) / sigma

    return {
        "z_score": z
    }

# ==============================
# 解析設定
# ==============================
num_runs = 5
num_surr = 39  # サロゲート数

# 欠損率は変化させない
missing_rates = [0.0]


# ==============================
# Bhatta_20フォルダ内のCSVを取得
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Bhatta_20")

if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"{data_dir} が見つかりません")

csv_files = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.lower().endswith(".csv")
])

print(f"{len(csv_files)} 個のCSVを検出しました。")

datasets = {}

for csv_path in csv_files:

    name = os.path.splitext(
        os.path.basename(csv_path)
    )[0]

    print(f"Loading: {name}")

    datasets[name] = load_csv_timeseries(csv_path)


print("\n読み込んだファイル")

for name in datasets:
    print(name)


# 全解析結果
all_results = []


# ==============================
# メイン解析
# ==============================
for name, base_data in datasets.items():

    # ファイル保存用の安全な名前
    file_safe_name = (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
    )

    # データごとの保存フォルダ
    system_dir = os.path.join(
        result_dir,
        file_safe_name
    )

    os.makedirs(system_dir, exist_ok=True)


    # ==============================
    # 同じデータを5回解析
    # ==============================
    for run in range(1, num_runs + 1):

        print("\n================================")
        print(f"Data : {name}")
        print(f"Run  : {run}/{num_runs}")
        print("================================")

        # runごとの保存フォルダ
        run_dir = os.path.join(
            system_dir,
            f"run_{run}"
        )

        os.makedirs(run_dir, exist_ok=True)


        # ==============================
        # 遅れ時間 tau 推定
        # ==============================
        tau_est = determine_tau(base_data)

        if tau_est < 1:
            tau_est = 1

        if tau_est > 50:
            tau_est = 50


        # ==============================
        # 埋め込み次元 m 推定
        # ==============================
        e1_values = itho_e1(
            base_data,
            max_dim=10,
            tau=tau_est
        )

        m_est = 2

        for i in range(1, len(e1_values)):

            if e1_values[i - 1] == 0:
                continue

            ratio = (
                e1_values[i]
                / e1_values[i - 1]
            )

            if abs(ratio - 1.0) < 0.05:
                m_est = i + 1
                break

        print(
            f"Estimated Parameters: "
            f"tau={tau_est}, "
            f"m={m_est}"
        )


        # ==============================
        # 欠損率ごとの処理
        # ==============================
        for rate in missing_rates:

            rate_dir = os.path.join(
                run_dir,
                f"rate_{int(rate * 100)}"
            )

            os.makedirs(
                rate_dir,
                exist_ok=True
            )


            # 欠損率0%の場合はpattern 1のみ
            if rate == 0.0:
                pattern_range = [0]
            else:
                pattern_range = range(3)


            patterns = {
                0: "random",
                1: "high",
                2: "low"
            }


            # ==============================
            # 欠損パターンごとの処理
            # ==============================
            for pattern_id in pattern_range:

                print(
                    f"\n--- Analyzing {name} "
                    f"Run={run}, "
                    f"Missing Rate={int(rate * 100)}%, "
                    f"Pattern={patterns[pattern_id]} ---"
                )

                pattern_dir = os.path.join(
                    rate_dir,
                    f"pattern_{pattern_id + 1}"
                )

                os.makedirs(
                    pattern_dir,
                    exist_ok=True
                )


                # ==============================
                # 欠損・線形補間
                # ==============================
                missing_data, data, mask = (
                    introduce_block_missing_and_interpolate(
                        base_data,
                        missing_rate=rate,
                        block_len=max(
                            10,
                            int(
                                len(base_data)
                                * rate
                                / 200
                            )
                        ),
                        seed=pattern_id,
                        pattern=patterns[pattern_id]
                    )
                )


                # ==============================
                # 時系列データ保存
                # ==============================
                ts_df = pd.DataFrame({
                    "time": np.arange(len(base_data)),
                    "original": base_data,
                    "missing": missing_data,
                    "interpolated": data,
                    "mask": mask.astype(int)
                })

                csv_path = os.path.join(
                    pattern_dir,
                    f"timeseries_{file_safe_name}.csv"
                )

                ts_df.to_csv(
                    csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(f"Saved CSV: {csv_path}")


                # ==============================
                # 欠損位置の可視化
                # ==============================
                if rate > 0:

                    plt.figure(figsize=(14, 4))

                    x_axis = np.arange(
                        len(base_data)
                    )

                    plt.plot(
                        x_axis,
                        base_data,
                        linewidth=1,
                        label="Original Data"
                    )

                    interp_only = np.full_like(
                        base_data,
                        np.nan,
                        dtype=float
                    )

                    interp_only[mask] = data[mask]

                    plt.plot(
                        x_axis,
                        interp_only,
                        linewidth=2,
                        label="Interpolated Segment"
                    )

                    plt.title(
                        f"{name} Missing Visualization\n"
                        f"Run={run}, "
                        f"Rate={int(rate * 100)}%, "
                        f"Pattern={patterns[pattern_id]}"
                    )

                    plt.xlabel("Time")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(alpha=0.3)

                    vis_path = os.path.join(
                        pattern_dir,
                        f"missing_vis_{file_safe_name}.png"
                    )

                    plt.savefig(
                        vis_path,
                        bbox_inches="tight"
                    )

                    print(
                        f"Saved visualization: "
                        f"{vis_path}"
                    )

                    plt.close()


                tau_val = tau_est

                print(
                    f">> Estimated Parameters: "
                    f"Tau={tau_val}, "
                    f"m={m_est}"
                )


                # ==============================
                # 元データのリアプノフ指数
                # ==============================
                print(
                    "Calculating original "
                    "Lyapunov exponent "
                    f"(m={m_est}, tau={tau_val})..."
                )

                real_lam = sano_sawada_lyapunov(
                    data,
                    m=m_est,
                    tau=tau_val,
                    theiler=tau_val * m_est
                )

                print(
                    f"Estimated m: {m_est} "
                    f"(E1 values: "
                    f"{e1_values[:m_est]})"
                )


                # ==============================
                # PPSサロゲート
                # ==============================
                print("Calculating PPS Surrogates...")

                pps_lams = []

                for k in range(num_surr):

                    print(
                        f"Run {run}: "
                        f"surrogate "
                        f"{k + 1}/{num_surr}"
                    )

                    surr = pps_surrogate(
                        data,
                        m_est,
                        tau=tau_val
                    )

                    lam = sano_sawada_lyapunov(
                        surr,
                        m=m_est,
                        tau=tau_val,
                        theiler=tau_val * m_est
                    )

                    pps_lams.append(lam)


                pps_lams = np.asarray(
                    pps_lams,
                    dtype=float
                )


                # ==============================
                # p値計算
                # ==============================
                rank = np.sum(
                    pps_lams >= real_lam
                )

                p_value = (
                    rank + 1
                ) / (
                    len(pps_lams) + 1
                )


                # ==============================
                # Zスコア計算
                # ==============================
                pps_result = surrogate_z_test(
                    real_lam,
                    pps_lams
                )


                # ==============================
                # サロゲート値の個別保存
                # ==============================
                result_df = pd.DataFrame({
                    "type": (
                        ["original"]
                        + ["PPS_surrogate"]
                        * num_surr
                    ),
                    "lambda": (
                        [real_lam]
                        + pps_lams.tolist()
                    )
                })

                result_csv_path = os.path.join(
                    pattern_dir,
                    f"lyapunov_results_"
                    f"{file_safe_name}.csv"
                )

                result_df.to_csv(
                    result_csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"Saved result CSV: "
                    f"{result_csv_path}"
                )


                # ==============================
                # Summaryへ追加
                # ==============================
                all_results.append({

                    "Data": name,
                    "Run": run,

  
                    "Original λ": round(
                        real_lam,
                        6
                    ),

                    "PPS Mean": round(
                        np.mean(pps_lams),
                        6
                    ),

                    "PPS Std": round(
                        np.std(
                            pps_lams,
                            ddof=1
                        ),
                        6
                    ),

                    "PPS Z-score": round(
                        pps_result["z_score"],
                        3
                    ),

                    "PPS Significance":
                        significance_label(
                            pps_result["z_score"]
                        ),

                    "P-value": round(
                        p_value,
                        4
                    )
                })


                # ==============================
                # Summaryを逐次保存
                # ==============================
                summary_df = pd.DataFrame(
                    all_results
                )

                summary_path = os.path.join(
                    result_dir,
                    "surrogate_summary.csv"
                )

                summary_df.to_csv(
                    summary_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"Updated Summary: "
                    f"{summary_path}"
                )


                # ==============================
                # ヒストグラム作成
                # ==============================
                plt.figure(figsize=(12, 8))

                plt.hist(
                    pps_lams,
                    bins=15,
                    alpha=0.7,
                    edgecolor="blue",
                    label="PPS Surrogates"
                )

                plt.axvline(
                    real_lam,
                    linestyle="--",
                    linewidth=3,
                    label=(
                        rf"Original "
                        rf"($\lambda$="
                        rf"{real_lam:.3f})"
                    )
                )

                plt.title(
                    f"{name}\n"
                    f"Run={run}/{num_runs}\n"
                    f"Missing={int(rate * 100)}%\n"
                    f"Pattern="
                    f"{patterns[pattern_id]}\n"
                    f"PPS test, "
                    f"p={p_value:.4f}"
                )

                plt.xlabel(
                    "Maximum Lyapunov "
                    "Exponent ($\\lambda$)"
                )

                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(
                    axis="y",
                    alpha=0.3
                )

                save_path = os.path.join(
                    pattern_dir,
                    f"result_{file_safe_name}.png"
                )

                plt.savefig(
                    save_path,
                    bbox_inches="tight"
                )

                print(f"Saved: {save_path}")

                plt.close()


                # ==============================
                # 結果表示
                # ==============================
                print(f"Original λ = {real_lam}")
                print(
                    f"PPS mean = "
                    f"{np.mean(pps_lams)}"
                )
                print(
                    f"PPS std = "
                    f"{np.std(pps_lams, ddof=1)}"
                )
                print(
                    f"PPS min = "
                    f"{np.min(pps_lams)}"
                )
                print(
                    f"PPS max = "
                    f"{np.max(pps_lams)}"
                )
                print(
                    f"PPS Z-score = "
                    f"{pps_result['z_score']}"
                )
                print(
                    f"Significance = "
                    f"{significance_label(pps_result['z_score'])}"
                )


# ==============================
# 最終Summary保存
# ==============================
summary_df = pd.DataFrame(all_results)

summary_path = os.path.join(
    result_dir,
    "surrogate_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved Final Summary: {summary_path}")