# 2026/05/04
#既存の力学系：ローレンツアトラク、ロジスティックス写像、正弦波、ホワイトノイズ、一次元ブラウン運動
#既存の力学系に対して解析を施しているがsurrogate.pyとは違い破損と線形補間を施している
#線形補間の影響が非線形性に対してどれだけ影響を及ぼすのか確認できる
#サロゲートデータ法の実装に関するプログラム
#生成法はランダムシャッフル(RS)とフーリエ変換(FT)
#統計量として佐野、沢田法(1987)に基づいてリアプノフ指数を使用している
#検定方法はモンテカルロ有意性検定
#佐野、沢田法において遅れ時間と埋め込み次元推定する必要がある
#遅れ時間は相互情報量、埋め込み次元は伊藤法E1で推定
#指標の平均を取るように変更



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

result_dir = "result_linear_avg"
if os.path.exists(result_dir):
    print(f"Cleaning up {result_dir}...")
    shutil.rmtree(result_dir)
    os.makedirs(result_dir)


# --- 力学系を生成 ---
def generate_brownian(n_steps=10000, D=1.0, dt=1.0):
    """
    1次元ブラウン運動（ランダムウォーク）
    x[0] = 0 から始めて、ガウス乱数を積分していく。
    D: 拡散係数
    dt: 時間刻み（ここでは便宜的に 1.0）
    """
    x = np.zeros(n_steps)
    # 増分 dW ~ N(0, 2 D dt)
    sqrt_2Ddt = np.sqrt(2.0 * D * dt)
    dW = sqrt_2Ddt * np.random.randn(n_steps - 1)
    x[1:] = np.cumsum(dW)
    return x

def generate_lorenz(n_steps=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    for i in range(n_steps-1):
        dx = sigma*(y[i]-x[i])
        dy = x[i]*(rho - z[i]) - y[i]
        dz = x[i]*y[i] - beta*z[i]
        x[i+1] = x[i] + dx*dt
        y[i+1] = y[i] + dy*dt
        z[i+1] = z[i] + dz*dt
    return x

def generate_sin(n_steps=10000, freq=1.0, dt=0.01):
    t = np.arange(n_steps) * dt
    return np.sin(2*np.pi*freq*t)

def generate_logistic(n_steps=10000, r=4.0, x0=0.3, discard=1000):
    x = np.zeros(n_steps+discard)
    x[0] = x0
    for i in range(n_steps+discard-1):
        x[i+1] = r*x[i]*(1-x[i])
    return x[discard:]

def generate_white_noise(n_steps=10000):
    return np.random.normal(0, 1, n_steps)


# ====== 破損線形補間を施している ======
def introduce_block_missing_and_interpolate(x, missing_rate=0.0, block_len=10, seed=0):
    """
    連続ブロック欠損をランダムに付与し、線形補間で埋める。
    missing_rate : 全体の何割を欠損させるか (0 <= r < 1)
    block_len    : 1ブロックの欠損長（サンプル数）
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if missing_rate <= 0:
        return x.copy(), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)

    total_missing = int(round(missing_rate * n))
    num_blocks = max(1, total_missing // block_len)

    for _ in range(num_blocks):
        start = rng.integers(0, n - block_len + 1)
        mask[start:start + block_len] = True

    # 上限調整
    if mask.sum() > total_missing:
        extra = mask.sum() - total_missing
        idx_true = np.where(mask)[0]
        idx_keep = rng.choice(idx_true, size=mask.sum() - extra, replace=False)
        mask[:] = False
        mask[idx_keep] = True

    y = x.copy()
    y[mask] = np.nan

    # 線形補間（端点は最近傍値で補完）
    xi = np.arange(n)
    valid = ~np.isnan(y)
    y_filled = np.interp(xi, xi[valid], y[valid])

    return y_filled, mask

def calculate_autocorrelation(series, max_lag):
    s = np.asarray(series, dtype=float)
    mean = np.mean(s)
    denom = np.sum((s - mean)**2)
    vals = []
    for lag in range(1, max_lag + 1):
        if lag >= len(s): break
        num = np.sum((s[:-lag] - mean) * (s[lag:] - mean))
        vals.append(num / denom if denom != 0 else 0.0)
    return vals

def determine_tau(series, max_lag=100):
    ac = calculate_autocorrelation(series, max_lag)
    for i, v in enumerate(ac):
        if v < 1/np.e: return i + 1
    return max_lag

def _embed(x, m, tau):
    N = len(x) - (m - 1)*tau
    if N <= 0: return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau*np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    if s is None: s = tau
    if k is None: k = max(1, int(0.05*len(x)))
    E1 = []
    for m in range(1, max_dim+1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0: break
        valid_M = M - s
        X_now = Xm[:valid_M]
        k_eff = min(k, valid_M-1)
        if k_eff < 1: break
        nn = NearestNeighbors(n_neighbors=min(k_eff+21, valid_M)).fit(X_now)
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
def sano_sawada_lyapunov(data, m = 3, tau = 1, n_neighbors = 30):
    
    N = len(data)
    #アトラクタの再構築
    embedded = np.array([data[i:N - (m-1)*tau + i:tau] for i in range(m)]).T
    num_points = len(embedded)

    num_points = len(embedded)
    step = tau
    theiler = tau * m

    lyap_exp = []
    
    for i in range(num_points - step):
        diff = embedded - embedded[i]
        dist = np.linalg.norm(diff, axis=1)
        
        sorted_idx = np.argsort(dist)

        #-- Theiler window--
        neighbors = []
        for j in sorted_idx[1:]:
            if abs(j - i) > theiler and (j + step < num_points):
                neighbors.append(j)
            if len(neighbors) >= n_neighbors:
                break

        if len(neighbors) < m:
            continue

        neighbors = np.array(neighbors)

        #ヤコビ行列の推定用行列構成
        z = embedded[neighbors] - embedded[i]
        z_next = embedded[neighbors + step] -embedded[i + step]

        try:
            #最小二乗法で A (z_next = A * z)を推定
            A, _, _, _ = np.linalg.lstsq(z, z_next, rcond=None)
            #特異値分解から最大伸長率を抽出
            singular_values = np.linalg.svd(A, compute_uv=False)
            if singular_values[0] > 1e-12:
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

# --- 解析の実行 ---
datasets = {
    "Brown motion": generate_brownian(),
    "Lorenz (Chaos)": generate_lorenz(),
    "Logistic (Chaos)": generate_logistic(),
    "Sine wave (Linear)": generate_sin(),
    "White Noise": generate_white_noise()
}

missing_rates = [0.0,0.1, 0.3, 0.5, 0.7, 0.9] #破損率10%~90%まで変化

num_trials = 5 #検定の平均を取るための試行回数
num_surr = 39 #有意水準2.5%用の39個

for name, base_data in datasets.items():
    for rate in missing_rates:
        print(f"\n{'='*30}")
        print(f"\n--- Analyzing {name} with Missing Rate {int(rate*100)}% ({num_trials} trials)")
        print(f"{'='*30}")

        trial_real_lams = []
        trial_ft_lams_all = []
        trial_rs_lams_all = []

        for t in range(num_trials):

            #データを破損させて、線形補間で埋める処理
            data, mask = introduce_block_missing_and_interpolate(
                base_data, missing_rate=rate, block_len=10, seed = t +100)

            # 遅れ時間　tau 
            tau_est = determine_tau(data, max_lag=100)
            tau_val = max(1, min(tau_est, 50))
            print(f"Estimated Tau:{tau_est}")

            #遅れ時間　m
            e1_values = itho_e1(data, max_dim=10, tau=tau_est)
            m_est = 3
            for i in range(1, len(e1_values)):
                if e1_values[i] > 0.8 and (e1_values[i] - e1_values[i-1]) < 0.02:
                    m_est = i + 1
                    break
            
            real_lam = sano_sawada_lyapunov(data, m=m_est, tau=tau_val)

            # 【追加】推定結果をメイン画面に表示
            print(f">> Estimated Parameters: Tau = {tau_val}, m = {m_est}")

            # 推定した値を使ってリアプノフ指数を計算
            print(f"Calculating original Lyapunov exponent (m={m_est}, tau={tau_val})......")
            trial_real_lams.append(real_lam)

            print(f"Calculating FT Surrogates......")
            ft_lams = [sano_sawada_lyapunov(ftsurrogate(data), m=m_est, tau=tau_val) for _ in range(num_surr)]


            print("Calculating Shuffle Surrogates......")
            rs_lams = [sano_sawada_lyapunov(rssurrogate(data), m=m_est, tau=tau_val) for _ in range(num_surr)]

            trial_ft_lams_all.extend(ft_lams)
            trial_rs_lams_all.extend(rs_lams)

            print(f"Estimated m: {m_est} (E1 values: {e1_values[:m_est]})")
            print(f"Trial {t+1}/{num_trials}: Original Lam = {real_lam:.4f}, Surr Mean = {np.mean(rs_lams):.4f}")

            avg_real_lams = np.mean(trial_real_lams)
            std_real_lams = np.std(trial_real_lams)

            #個別のグラフの作成
            plt.figure(figsize=(10, 6))

            #ヒストグラムを２つ重ねで表示
            plt.hist(trial_ft_lams_all, bins=25, color='skyblue', alpha=0.6,
                    label=f'FT Surrogates (N={len(trial_ft_lams_all)})', edgecolor='blue')
            plt.hist(trial_rs_lams_all, bins=25, color='orange', alpha=0.6,
                    label=f'Shuffle Surrogates (N={len(trial_rs_lams_all)})', edgecolor='red')
            
            #オリジナルの値を赤線で表示
            plt.axvline(avg_real_lams, color='red', linestyle='--', linewidth=3,
                        label=f'Avg Original ($\lambda$={real_lam:.3f})')
            plt.axvspan(avg_real_lams - std_real_lams, avg_real_lams + std_real_lams,
                        color='red', alpha=0.2, label='Original StdDev')
            
            plt.title(f"{name} (Missing Rate: {int(rate*100)}%, {num_trials} Trial Avg)")
            plt.xlabel("Maximum Lyapunov Exponent ($\lambda$)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

            # 【追加】resultフォルダにファイルとして保存
            file_safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(result_dir, f"result_{file_safe_name}_rate{int(rate*100)}.png")
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
            
            # メモリ節約とグラフ重複防止のため閉じる
            plt.close()
    
plt.show()


