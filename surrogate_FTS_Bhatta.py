# 2026/05/18
#線形補間の影響が非線形性に対してどれだけ影響を及ぼすのか確認できる
#サロゲートデータ法の実装に関するプログラム
#生成法はランダムシャッフル(RS)とフーリエ変換(FT)
#統計量として佐野、沢田法(1987)に基づいてリアプノフ指数を使用している
#検定方法はモンテカルロ有意性検定
#遅れ時間は相互情報量、埋め込み次元は伊藤法E1で推定
#欠損の方法として欠損部位をブロックごとランダムに変更（pattern 1~3)
#Bhatta 20個のデータを処理

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

result_dir = "result_FT&S_Bhatta"
if os.path.exists(result_dir):
    print(f"Cleaning up {result_dir}...")
    shutil.rmtree(result_dir)
    os.makedirs(result_dir)

def load_csv_timeseries(csv_path):

    df = pd.read_csv(
        csv_path,
        header=None
    )

    return df.iloc[:, 0].astype(float).values


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

# --- サロゲートデータ生成　（FT法）---
def ftsurrogate(x):
    x = np.asarray(x)
    y = np.fft.fft(x)
    n = len(x)

    if n % 2 == 0:
        l = n // 2 - 1
        r = np.exp(2j * np.pi * np.random.rand(l))
        v = np.concatenate(([1], r, [1], np.flip(np.conj(r))))
    else:
        l = (n - 1) // 2
        r = np.exp(2j * np.pi * np.random.rand(l))
        v = np.concatenate(([1], r, np.flip(np.conj(r))))

    z = np.fft.ifft(y * v)
    return np.real(z)

# --- 追加：ランダムシャッフルサロゲートでの検定 ---
def rssurrogate(x):
    x_random = np.random.permutation(x)

    #Calculating and plotting autocorrelation
    #autocorrelation_plot(pd.Series(x))
    #autocorrelation_plot(pd.Series(x_random))
    return np.array(x_random)

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
num_surr = 39   # 有意水準2.5%用

# ==============================
# Bhatta20フォルダ内のCSVを自動取得
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

    name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"Loading : {name}")

    datasets[name] = load_csv_timeseries(csv_path)

all_results = []

# ==============================
# メイン解析
# ==============================
for name, base_data in datasets.items():

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

    for run in range(1, num_runs + 1):

        print("\n==============================")
        print(f"Dataset : {name}")
        print(f"Run     : {run}/{num_runs}")
        print("==============================")

        # runごとの保存フォルダ
        run_dir = os.path.join(
            system_dir,
            f"run_{run}"
        )
        os.makedirs(run_dir, exist_ok=True)

        # 欠損は加えない
        data = base_data.copy()

        # ===== 時系列データ保存 =====
        ts_df = pd.DataFrame({
            "time": np.arange(len(base_data)),
            "data": data
        })

        csv_path = os.path.join(
            run_dir,
            f"timeseries_{file_safe_name}.csv"
        )

        ts_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # ===== 遅れ時間 tau 推定 =====
        tau_est = determine_tau(data, max_lag=100)

        if tau_est < 1:
            tau_est = 1

        if tau_est > 50:
            tau_est = 50

        tau_val = tau_est

        print(f"Estimated Tau: {tau_val}")

        # ===== 埋め込み次元 m 推定 =====
        e1_values = itho_e1(
            data,
            max_dim=10,
            tau=tau_val,
            theiler=tau_val * 2
        )

        m_est = 2

        for i in range(1, len(e1_values)):

            ratio = e1_values[i] / e1_values[i - 1]

            if abs(ratio - 1.0) < 0.05:
                m_est = i + 1
                break

        print(f">> Estimated Parameters: Tau = {tau_val}, m = {m_est}")
        print(f"Estimated m: {m_est} (E1 values: {e1_values[:m_est]})")

        # ===== オリジナルデータのリアプノフ指数 =====
        print(
            f"Calculating original Lyapunov exponent "
            f"(m={m_est}, tau={tau_val})..."
        )

        real_lam = sano_sawada_lyapunov(
            data,
            m=m_est,
            tau=tau_val,
            theiler=tau_val * m_est
        )

        # ===== FTサロゲート =====
        print("Calculating FT Surrogates...")

        ft_lams = [
            sano_sawada_lyapunov(
                ftsurrogate(data),
                m=m_est,
                tau=tau_val,
                theiler=tau_val * m_est
            )
            for _ in range(num_surr)
        ]

        # ===== ランダムシャッフルサロゲート =====
        print("Calculating Shuffle Surrogates...")

        rs_lams = [
            sano_sawada_lyapunov(
                rssurrogate(data),
                m=m_est,
                tau=tau_val,
                theiler=tau_val * m_est
            )
            for _ in range(num_surr)
        ]

        # ===== 結果CSV保存 =====
        result_df = pd.DataFrame({
            "type": (
                ["original"] +
                ["FT_surrogate"] * num_surr +
                ["Shuffle_surrogate"] * num_surr
            ),
            "lambda": (
                [real_lam] +
                ft_lams +
                rs_lams
            )
        })

        result_csv_path = os.path.join(
            run_dir,
            f"lyapunov_results_{file_safe_name}.csv"
        )

        result_df.to_csv(result_csv_path, index=False)
        print(f"Saved result CSV: {result_csv_path}")

    # ===== Z-score calculation =====
        ft_result = surrogate_z_test(real_lam, ft_lams)
        rs_result = surrogate_z_test(real_lam, rs_lams)

        all_results.append({

                "Data": name,
                "Original λ": round(real_lam, 6),

                "FT Z-score": round(ft_result["z_score"], 3),
                "FT Significance": significance_label(ft_result["z_score"]),

                "RS Z-score": round(rs_result["z_score"], 3),
                "RS Significance": significance_label(rs_result["z_score"]),

        })



        # ===== ヒストグラム作成 =====
        plt.figure(figsize=(10, 6))

        plt.hist(
            ft_lams,
            bins=15,
            color='skyblue',
            alpha=0.6,
            label='FT Surrogates (Phase Random)',
            edgecolor='blue'
        )

        plt.hist(
            rs_lams,
            bins=15,
            color='orange',
            alpha=0.6,
            label='Shuffle Surrogates (ALL Random)',
            edgecolor='red'
        )

        plt.axvline(
            real_lam,
            color='red',
            linestyle='--',
            linewidth=3,
            label=rf'Original ($\lambda$={real_lam:.3f})'
        )

        plt.title(
            f"{name} "
            f"(Run {run}/{num_runs})"
        )

        plt.xlabel("Maximum Lyapunov Exponent ($\\lambda$)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        save_path = os.path.join(
            run_dir,
            f"result_{file_safe_name}.png"
        )

        plt.savefig(save_path)
        print(f"Saved figure: {save_path}")

        plt.close()

summary_df = pd.DataFrame(all_results)

summary_path = os.path.join(
     result_dir,
        "surrogate_summary.csv"
    )

summary_df.to_csv(
     summary_path,
       index=False
    )

print(f"Saved Summary: {summary_path}")

plt.show()
