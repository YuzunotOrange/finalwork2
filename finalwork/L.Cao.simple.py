#2025/08/28
#Yuzu Matsumoto
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threshold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1
    return max_lag

def estimate_min_dimension(E1, eps=0.01, win = 2):
    if len(E1) <= 1: return len(E1)
    diff = np.abs(np.diff(E1))
    for m in range(len(diff) - win + 1):
        if np.all(diff[m:m+win] < eps):
            return m + 1
    return len(E1)

def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

#cao法
def cao_e1_e2(x, max_dim = 10, tau = 1):
    E1 = []
    for m in range(1, max_dim + 1):
        if (m + 1) * tau > len(x): break
        Xm = _embed(x, m, tau)
        Xm1 = _embed(x, m + 1, tau)

        #Xm と　Xm1の行列を揃える
        N = min(len(Xm), len(Xm1))
        Xm = Xm[:N]
        Xm1 = Xm1[:N]


        d = np.linalg.norm(Xm[:, None, :] - Xm[None, :, :], axis = 2)
        np.fill_diagonal(d, np.inf)
        j = d.argmin(axis = 1)
        base = np.linalg.norm(Xm - Xm[j], axis = 1)
        ext = np.linalg.norm(Xm1 - Xm1[j], axis = 1)
        E1.append(np.mean(ext / base))
    return np.array(E1), None

#伊藤法
def itho_e1_e2(x, max_dim = 10, tau = 1, s = 1, k_frac = 0.05):
    E1 = []
    N = len(x)
    k = max(1, int(N * k_frac))
    for m in range(1, max_dim + 1):
        if (m + 1) * tau > len(x): break
        Xm = _embed(x, m, tau)
        dists = np.linalg.norm(Xm[:, None, :] - Xm[None, :, :], axis = 2)
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis = 1)
        E1.append(np.mean(sorted_dists[:, :k]))
    return np.array(E1), None

tau = determine_tau(x, max_lag= auto_max_lag)

#Cao法
E1_c, _ = cao_e1_e2(x, max_dim = args.max_dim, tau = tau)
dmin_c = estimate_min_dimension(E1_c)

#伊藤法
E1_i, _ = itho_e1_e2(x, max_dim = args.max_dim, tau = tau, s = tau, k_frac = auto_k_frac)
dmin_i = estimate_min_dimension(E1_i)

#出力結果
print("length of data:", N)
print("auto setting max_lag:", auto_max_lag)
print("auto setting k_frac:", auto_k_frac)
print("Cao d_min:", dmin_c)
print("Itho d_min:", dmin_i)

#plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(E1_c) + 1), E1_c, label = 'Cao E1')
plt.plot(range(1, len(E1_i) + 1), E1_i, label = "Itho E1")
plt.axvline(dmin_c, color = 'blue', linestyle = '--', label = f'Cao d_min = {dmin_c}')
plt.axvline(dmin_i, color = 'red', linestyle = '--', label = f"Itho d_min = {dmin_i}")
plt.xlabel("Embedding Dimension")
plt.ylabel("E1 Value")
plt.title("Minimum Embedding Dimension Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()