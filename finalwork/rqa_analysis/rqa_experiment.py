#2025-10-01
#RQAを実装するためのプログラム

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#from pyunicorn.timeseries import RecurrencePlot
#import nolds
def load_first_column(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] < 1:
        raise ValueError("threr is no data in input file")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    x = s.to_numpy(dtype=float) 
    mu = np.mean(x)
    sd = np.std(x)
    return(x - mu) / sd if sd > 0 else (x - mu)

### 2. Estimation embedding dimension m (way of Cao) 埋め込み次元　m 推定　(Caoの方法)
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

###Estimation of delay time　τ　(Mutual Information)　遅延時間　推定　(相互情報量)
def determine_tau(series, max_lag=100):
    autocorr = calculate_autocorrelation(series, max_lag)
    threshold = 1 / np.e
    for i, val in enumerate(autocorr):
        if val < threshold:
            return i + 1
    return max_lag

def _embed(series, m, tau):
    series = np.asarray(series, dtype=float)
    N = len(series) - (m - 1) * tau
    if N <= 1:
        raise ValueError("Series too short for embedding with given m, tau.")
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return series[idx]

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

#---RQA用----
def load_series_first_col(path: str) -> np.ndarray:
    base = os.path.basename(path).lower()
    if base == "lorez.csv":
        df = pd.read_csv(path, header=None, sep=r"\s+") #空白区切り
    else:
        #区切り自動推定
        df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] < 1:
        raise ValueError("There is no column")
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if s.isna().any():
        s = s.interpolate(limit_direction="both").bfill().ffill()
    x = s.to_numpy(dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / sd if sd > 0 else (x - mu)

def recurrence_matrix(X: np.ndarray, epsilon: float, normalize: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xn = (X / np.std(X)) if (normalize and np.std(X) > 0) else X
    D = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(axis=2))
    R = (D <= epsilon).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

def diag_line_length(R: np.ndarray) -> np.ndarray:
    N = R.shape[0]
    lengths = []
    for k in range(-(N - 1), N):
        d = np.diag(R, k=k)
        if d.size == 0:
            continue
        run = 0
        for v in d:
            if v == 1:
                run += 1
            else:
                if run > 0:
                    lengths.append(run); run = 0
        if run > 0:
            lengths.append(run)
    return np.array(lengths, dtype=int) if lengths else np.array([], dtypr=int)

def rqa_measures(R: np.ndarray, l_min: int = 2):
    lengths = diag_line_length(R)
    RR_points = int(R.sum())
    if lengths.size == 0:
        return 0.0, 0.0, 0.0
    sel = lengths[lengths >= l_min]
    if sel.size == 0 or RR_points == 0:
        return 0.0, 0.0, 0.0
    det = float(np.sum(sel) / RR_points)
    L = float(np.mean(sel))
    unique, counts = np.unique(sel, return_counts=True)
    p = counts / counts.sum()
    EN = float(-np.sum(p * np.log(p + 1e-12)))
    return det, L, EN

def epsilon_for_target_rr(X: np.ndarray, target_rr: float = 0.05, normalize: bool = True) -> float:
    """
    目標再帰率（RR）になるように ε を距離分布の分位点から決める。
    """
    X = np.asarray(X, float)
    if normalize and np.std(X) > 0:
        X = X / np.std(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    flat = D[np.isfinite(D)]
    k = max(1, int(target_rr * flat.size))
    return float(np.partition(flat, k)[k])

##ここ以降はmainです。 2025-10-01
def main():
    parser = argparse.ArgumentParser(description="RQA pipeline using fixed function (first column only)")
    parser.add_argument("csv_path", help="input file pass (ex, lorenz.csv)")
    parser = argparse.ArgumentParser(description="RQA for a single series with lorenz.csv integration")
    parser.add_argument("csv_path", nargs="?", help="入力CSVパス。省略時はスクリプト同階層の lorenz.csv を自動使用")
    parser.add_argument("--max-lag", type=int, default=200, help="τ推定の最大ラグ（既定: 200）")
    parser.add_argument("--max-dim", type=int, default=12, help="Cao(E1)の最大次元（既定: 12）")
    parser.add_argument("--lmin", type=int, default=2, help="RQAの最小対角線長（既定: 2）")
    parser.add_argument("--eps-min", type=float, default=0.1)
    parser.add_argument("--eps-max", type=float, default=2.0)
    parser.add_argument("--eps-steps", type=int, default=20)
    parser.add_argument("--plot-rp", action="store_true", help="リカレンスプロットも表示")
    args = parser.parse_args()
   
   # 入力ファイルの決定（省略時は lorenz.csv）
    if args.csv_path:
        csv_path = args.csv_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "lorenz.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {csv_path}")

    # 1) 読み込み（lorenz.csv は空白区切り・1列目のみ、それ以外は区切り自動）
    series = load_series_first_col(csv_path)


    #estimate τ
    tau = determine_tau(series, max_lag=args.max_lag)

    #estimate m
    E1 = itho_e1(series, max_dim=args.max_dim, tau=tau, s=tau, theiler=tau)
    m = estimate_min_dimension(E1, eps=0.01, win=3)

    #RQA
    X = _embed(series, m, tau)
    epsilons = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    det_list, l_list, entr_list = [], [], []
    for eps in epsilons:
        R = recurrence_matrix(X, epsilon=eps, normalize=True)
        det, L, EN = rqa_measures(R, l_min=2)
        det_list.append(det); l_list.append(L); entr_list.append(EN)

    ### 5. average of result and disply 結果の平均と表示
    mD = float(np.mean(det_list))
    mL = float(np.mean(l_list))
    mEN = float(np.mean(entr_list))

    print("\n==== RQA RESULT OF AVERAGE ====")
    print(f"File: {os.path.basename(csv_path)}")
    print(f"tau (delay time): {tau}")
    print(f"m (embedded dimension): {m}")
    print(f"Determinism (mD): {mD:.3f}")
    print(f"Line Length (mL): {mL:.3f}")
    print(f"Entropy (mEN): {mEN:.3f}")

    ### 6. Disply as glaph
    plt.figure(figsize=(10, 4))
    plt.plot(epsilons, det_list, label='Determinism')
    plt.plot(epsilons, l_list, label='Line Length')
    plt.plot(epsilons, entr_list, label='Entropy')
    plt.xlabel("Threshold  ε ")
    plt.ylabel("RQA Measure")
    plt.title("RQA Measures vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show
    
    # 7) リカレンスプロット（任意）
    if args.plot_rp:
        try:
            eps_rp = epsilon_for_target_rr(X, target_rr=0.05, normalize=True)
            R_rp   = recurrence_matrix(X, epsilon=eps_rp, normalize=True)
            plt.figure(figsize=(5,5))
            plt.imshow(R_rp, origin='lower', cmap='gray_r', interpolation='none')
            plt.title(f"Recurrence Plot (ε≈{eps_rp:.3g}, RR={R_rp.mean():.3f})")
            plt.xlabel("i"); plt.ylabel("j")
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[WARN] RP描画に失敗: {e}")

if __name__ == "__main__":
    # 例: python rqa_experiment.py lorenz.csv
    main()