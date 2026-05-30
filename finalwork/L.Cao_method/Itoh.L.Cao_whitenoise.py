#2025/09/02
#Yuzu Matsumoto
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

#---引数---
ap = argparse.ArgumentParser()
ap.add_argument("--csv_files", required=True, help="Whitespace-delimited text or CSV file (first column will be used)")
ap.add_argument("--max_dim", type=int, default=10, help="Maximum embedding dimension(default = 10)")
args = ap.parse_args()

#scan data
new_data = pd.read_csv(args.csv_files, header=None, delim_whitespace=True)
series = new_data.iloc[:, 0].values
x = series 

#自動パラメータ設定
N = len(x)
auto_max_lag = min(100, N // 10) #100以下に設定
auto_k_frac = min(0.1, 1 / np.sqrt(N)) #小さすぎない範囲


def calculate_autocorrelation(series, max_lag):
    autocorr_values = []
    series_mean = np.mean(series)
    denom = np.sum((series - series_mean) ** 2)
    for lag in range(1, max_lag + 1):
        if lag >= len(series): break
        num = np.sum((series[:-lag] - series_mean) * (series[lag:] - series_mean))
        autocorr_values.append((num / denom) if denom != 0 else 0.0)
    return autocorr_values


def estimate_min_dimension(E1, eps=0.05, win = 3):
    E1 = np.asarray(E1, dtype=float)
    if len(E1) <= 1: 
        return len(E1)
    for m in range(1, len(E1) - win + 1):
        seg = E1[m:m+win]
        dif = np.abs(np.diff(seg))
        base = np.maximum(np.abs(seg[:-1]), 1e-12)
        if np.all(dif / base <= eps):
            return m + 1
    return len(E1)

def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

#伊藤法
def itho_e1_e2(x, max_dim = 10, tau = 1, s = None, k = None, theiler = 0):
    N = len(x)
    
    if s is None: s = tau
    if k is None: k = max(1, int(0.05 * N))
    
    E1 = []
    for m in range(1, max_dim + 1):
        Xm = _embed(x, m, tau)
        M = Xm.shape[0]
        if M <= s or M == 0: #s ステップ先は作れない
            break

        #k　最近傍探索
        valid_M = M - s
        X_now = Xm[:valid_M]

        k_eff = min(k, valid_M - 1)
        over_k = min(valid_M - 1, k_eff + 20)

        nn = NearestNeighbors(n_neighbors= over_k+1, algorithm = 'auto', metric='euclidean')
        nn.fit(X_now)
        dists, indxs = nn.kneighbors(X_now) #自身を含む

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
    
    return np.array(E1)

def process_data(x, label, color):
    tau = 5

    E1_i = itho_e1_e2(x, max_dim=args.max_dim, tau = tau, s=tau, k=int(0.05*len(x)))
    dmin_i = estimate_min_dimension(E1_i, eps=0.05, win=3)

    
    print(f"[{label}] length of data:", len(x))
    print(f"[{label}] Itho d_min:", dmin_i)
    print(f"[{label}] tau:", tau, "k:", int(0.05*len(x)))
    print(f"[{label}] E1_i:", np.array2string(E1_i, precision=3))

    plt.plot(range(1, len(E1_i) + 1), E1_i, label=f"{label} E1", color=color)
    plt.axvline(dmin_i, color=color, linestyle="--", label=f"{label} d_min={dmin_i}")

#ノイズあり
x_noisy = x + np.random.uniform(0, 1, size=N)
#plot
plt.figure(figsize=(10, 5))
process_data(x, label="Original", color="blue")
process_data(x_noisy, label="Noisy", color="orange")
plt.xlabel("Embedding Dimension")
plt.ylabel("E1 Value")
plt.yscale('log')
plt.title("Minimum Embedding Dimension Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()