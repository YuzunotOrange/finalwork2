# 2026/04/15
#サロゲートデータ法の実装に関するプログラム
#生成法はランダムシャッフル(RS)とフーリエ変換(FT)
#統計量として佐野、沢田法(1987)に基づいてリアプノフ指数を使用している
#検定方法はモンテカルロ有意性検定
#佐野、沢田法において遅れ時間と埋め込み次元推定する必要があるが今回は固定

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import sys
import datetime
import random

# --- 力学系を生成 ---
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

# --- 佐野、沢田法（最大リアプノフ指数推定）---
def sano_sawada_lyapunov(data, m = 3, tau = 1, n_neighbors = 30):
    
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
        #近傍点のインデックスを取得
        nearest_idx = np.argsort(dist)[1:n_neighbors+1]

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

# --- 解析の実行 ---
datasets = {
    "Lorenz (Chaos)": generate_lorenz(),
    "Logistic (Chaos)": generate_logistic(),
    "Sine wave (Linear)": generate_sin(),
    "White Noise": generate_white_noise()
}

num_surr = 39 #有意水準2.5%用の39個

for name, data in datasets.items():
    print(f"\n--- Analyzing {name} ---")

    #連続系と離散系で tau を調整
    tau_val = 10 if "Lorenz" in name or "Sin" in name else 1

    #オリジナルの計算
    print(f"Caluculating original Lyapunov exponent......")
    real_lam = sano_sawada_lyapunov(data, m=3, tau=tau_val)

    #フーリエ変換サロゲートデータの計算
    print(f"Calculating FT Surrogates......")
    ft_lams = [sano_sawada_lyapunov(ftsurrogate(data), m=3, tau=tau_val) for _ in range(num_surr)]

    #ランダムシャッフルサロゲートデータの計算
    print("Calculating Shuffle Surrogates......")
    rs_lams = [sano_sawada_lyapunov(rssurrogate(data), m=3, tau=tau_val) for _ in range(num_surr)]

    #個別のグラフの作成
    plt.figure(figsize=(10, 6))

    #ヒストグラムを２つ重ねで表示
    plt.hist(ft_lams, bins=15, color='skyblue', alpha=0.6, label='FT Surrogates (Phase Random)', edgecolor='blue')
    plt.hist(rs_lams, bins=15, color='orange', alpha=0.6, label='Shuffle Surrogates (ALL Random)', edgecolor='red')
    
    #オリジナルの値を赤線で表示
    plt.axvline(real_lam, color='red', linestyle='--', linewidth=3, label=f'Original ($\lambda$={real_lam:.3f})')

    plt.title(f"Comparison of Surrogates: {name}")
    plt.xlabel("Maximum Lyapunov Exponent ($\lambda$)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
plt.show()


