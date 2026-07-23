#4指標それぞれの時系列を出力する

import numpy as np
import pandas as pd
import re, os
import shutil
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt   # ★ 追加

# ====== 関数群 ======
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

def estimate_min_dimension(E1, eps=0.05, win=3):
    E1 = np.asarray(E1)
    if E1.size == 0: return 1
    for m in range(1, len(E1)-win+1):
        seg = E1[m:m+win]
        if np.all(np.abs(np.diff(seg))/np.maximum(np.abs(seg[:-1]),1e-12) <= eps):
            return m+1
    return len(E1)

def sanitize_filename(name):
    return re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(name))[0])

def load_series_first_col_auto(path):
    try:
        df = pd.read_csv(path, header=None, usecols=[0], comment="#", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, header=None, sep=None, engine="python", comment="#", usecols=[0])
    s = pd.to_numeric(df.iloc[:,0], errors="coerce").interpolate().bfill().ffill()
    x = s.to_numpy(dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu)/sd if sd>0 else (x - mu)

def recurrence_matrix(X, eps, normalize=True):
    if normalize:
        sd = np.std(X)
        X = X / sd if sd>0 else X
    D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    R = (D <= eps).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

def diag_line_lengths(R):
    N = len(R)
    Ls = []
    for k in range(-N+1, N):
        diag = np.diag(R, k)
        run = 0
        for v in diag:
            if v == 1: run += 1
            elif run>0: Ls.append(run); run = 0
        if run>0: Ls.append(run)
    return np.array(Ls, int)

def vertical_line_lengths(R):
    N = len(R)
    Ls = []
    for j in range(N):
        col = R[:, j]
        run = 0
        for v in col:
            if v==1: run+=1
            elif run>0: Ls.append(run); run=0
        if run>0: Ls.append(run)
    return np.array(Ls, int)

def white_vertical_line_lengths(R):
    N = len(R)
    Ls = []
    for j in range(N):
        col = R[:, j]
        run = 0
        for v in col:
            if v==0: run+=1
            elif run>0: Ls.append(run); run=0
        if run>0: Ls.append(run)
    return np.array(Ls, int)

def entropy_from_lengths(Ls):
    if len(Ls)==0: return 0.0
    _, c = np.unique(Ls, return_counts=True)
    p = c / np.sum(c)
    return -np.sum(p*np.log(p+1e-12))

def rqa_measures(R, lmin=2):
    Ls = diag_line_lengths(R)
    RR = int(R.sum())
    if len(Ls)==0 or RR==0: return 0,0,0
    Ls = Ls[Ls>=lmin]
    if len(Ls)==0: return 0,0,0
    det = np.sum(Ls)/RR
    L = np.mean(Ls)
    EN = entropy_from_lengths(Ls)
    return det,L,EN

def rqa_measures_full(R, lmin=2, vmin=2, wmin=2):
    rr = float(R.mean())
    dls = diag_line_lengths(R)
    dsel = dls[dls >= lmin]
    det = np.sum(dsel)/R.sum() if len(dsel)>0 and R.sum()>0 else 0.0
    L = np.mean(dsel) if len(dsel)>0 else 0.0
    Lmax = np.max(dsel) if len(dsel)>0 else 0
    L_entr = entropy_from_lengths(dsel)
    DIV = 1/Lmax if Lmax>0 else 0.0

    vls = vertical_line_lengths(R)
    vsel = vls[vls >= vmin]
    LAM = np.sum(vsel)/R.sum() if len(vsel)>0 and R.sum()>0 else 0.0
    TT = np.mean(vsel) if len(vsel)>0 else 0.0

    wls = white_vertical_line_lengths(R)
    wsel = wls[wls >= wmin]
    W = np.mean(wsel) if len(wsel)>0 else 0.0
    Wmax = np.max(wsel) if len(wsel)>0 else 0
    W_div = 1/Wmax if Wmax>0 else 0.0
    DEratio = det/rr if rr > 0 else np.nan

    return {
        "RR": rr, "DET": det, "DEratio": DEratio, "L": L, "L_max": Lmax, "DIV": DIV, "L_entr": L_entr,
        "LAM": LAM, "TT": TT, "W": W, "W_max": Wmax, "W_div": W_div
    }

def epsilon_for_target_rr(X, target_rr=0.05):
    D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    flat = D[np.isfinite(D)]
    k = max(1, int(target_rr*len(flat))-1)
    return float(np.partition(flat, k)[k])

def compute_rqa_from_tau_m(series, tau, m, epsilons, lmin=2):
    X = _embed(series, m, tau)
    if X.size==0:
        return [],[],[],{"m_use":m,"mD":np.nan,"mL":np.nan,"mEN":np.nan},X
    dets,Ls,ENs = [],[],[]
    for e in epsilons:
        R = recurrence_matrix(X,e)
        d,L,EN = rqa_measures(R,lmin)
        dets.append(d); Ls.append(L); ENs.append(EN)
    return dets,Ls,ENs,{
        "m_use":m,
        "mD":np.mean(dets),
        "mL":np.mean(Ls),
        "mEN":np.mean(ENs)
    },X


# ===== 4系列生成 =====
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


# ====== 解析 ======
def analyze_series(name, series, max_dim=10, max_lag=100, target_rr=0.05,
                   max_points_for_rqa=2000):
    print(f"\n===== {name} =====")
    x = np.asarray(series, dtype=float)
    x = (x - x.mean()) / (x.std() if x.std() > 0 else 1)

    tau = determine_tau(x, max_lag=max_lag)
    print(f"tau = {tau}")

    E1 = itho_e1(x, max_dim=max_dim, tau=tau)
    m_est = estimate_min_dimension(E1, eps=0.05, win=3)
    print(f"estimated m = {m_est}")

    X_tmp = _embed(x, m_est, tau)

    # RQA に使う点数をサンプリングして減らす
    N = X_tmp.shape[0]
    if N > max_points_for_rqa:
        step = int(np.ceil(N / max_points_for_rqa))
        X_rqa = X_tmp[::step]
        print(f"Using {X_rqa.shape[0]} points out of {N} for RQA (step={step})")
    else:
        X_rqa = X_tmp

    # eps もサンプリング後のデータで計算
    eps = epsilon_for_target_rr(X_rqa, target_rr=target_rr)
    print(f"epsilon = {eps}")

    # 詳細版 RQA（軽くなったデータで）
    R = recurrence_matrix(X_rqa, eps)
    full = rqa_measures_full(R)

    result = {
        "name": name,
        "tau": tau,
        "min_m": m_est,
        "epsilon": eps,
    }
    result.update(full)
    return result


# ====== 4系列の「線＋点」図を保存 ======
import shutil
import matplotlib.pyplot as plt

def save_timeseries_plots(series_dict, out_dir="plots_timeseries",
                          max_points_for_plot=1000):
    """
    series_dict: {name: 1次元配列} の辞書
    各系列を Time vs Value の線＋点で描いて PNG 保存
    max_points_for_plot: 図に描く最大サンプル数（多いと真っ赤になるので制限）
    """
    # 既存フォルダがあれば削除して作り直す
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for name, s in series_dict.items():
        s = np.asarray(s)

        # ---- プロット用に間引き ----
        n = len(s)
        if n > max_points_for_plot:
            step = int(np.ceil(n / max_points_for_plot))
        else:
            step = 1

        t_plot = np.arange(0, n, step)
        s_plot = s[::step]

        plt.figure(figsize=(8, 4))

        # 赤い線（間引き後）
        plt.plot(t_plot, s_plot, "-", linewidth=1.0,
                 color="red", label="Series (line)")
        # 青い点（間引き後）
        plt.scatter(t_plot, s_plot, s=10,
                    color="blue", label="Series (points)")

        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.title(name)
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc="best")

        fname = sanitize_filename(name) + "_timeseries.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[Saved] {out_path}")



# ====== メイン ======
def main():
    n = 8000
    np.random.seed(0)

    series_dict = {
        "Lorenz attractor (x)": generate_lorenz(n_steps=n),
        "Sine wave":            generate_sin(n_steps=n),
        "Logistic map":         generate_logistic(n_steps=n),
        "White noise":          generate_white_noise(n_steps=n),
    }

    # 解析
    results = []
    for name, s in series_dict.items():
        res = analyze_series(name, s, max_dim=10, max_lag=200, target_rr=0.05)
        results.append(res)

    # CSV 保存
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False, encoding="utf-8")
    print("\nSaved: results.csv")

    # ★ 各系列の「線＋点」図を保存 ★
    save_timeseries_plots(series_dict, out_dir="plots_timeseries")


if __name__ == "__main__":
    main()
