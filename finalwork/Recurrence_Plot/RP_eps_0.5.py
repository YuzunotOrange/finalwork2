import argparse
import os
import re
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans



# ====== 設定 ======
MAX_DIM = 10
K_FRAC  = 0.50
MAX_LAG_TAU = 100
MAIN_OUT_DIR = "rqa_result_DEratio"
DEFAULT_INPUT_DIR = "new_li"
# 共通のグループカラー
GROUP_COLORS = {
    "Low":  plt.cm.tab20(0),  # 濃い青
    "Mid":  plt.cm.tab20(1),  # 薄い青
    "High": plt.cm.tab20(2),  # オレンジ
}

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
    if normalize: X = X / np.std(X) if np.std(X)>0 else X
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
    if X.size==0: return [],[],[],{"m_use":m,"mD":np.nan,"mL":np.nan,"mEN":np.nan},X
    dets,Ls,ENs = [],[],[]
    for e in epsilons:
        R = recurrence_matrix(X,e)
        d,L,EN = rqa_measures(R,lmin)
        dets.append(d); Ls.append(L); ENs.append(EN)
    return dets,Ls,ENs,{"m_use":m,"mD":np.mean(dets),"mL":np.mean(Ls),"mEN":np.mean(ENs)},X

def main():
    parser = argparse.ArgumentParser(
        description="指定データ(1列目)からε=0.5のリカレンスプロットを作成するスクリプト"
    )

    parser.add_argument("input", help = "入力データファイル")
    parser.add_argument(
        "--eps", type=float, default=0.5,
        help="リカレンスプロットの閾値ε"
    )
    parser.add_argument(
        "--tau", type=int, default=None,
        help="遅れ時間 tau を手動で設定（未指定であった場合、自動で推定）"
    )
    parser.add_argument(
        "--m", type=int, default=None,
        help="埋め込み次元ｍを自動で推定(未指定であった場合は自動で推定する)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="出力画像ファイル"
    )

    args = parser.parse_args()

    #データの読み込み
    series = load_series_first_col_auto(args.input)

    #tau決定
    if args.tau is None:
        tau = determine_tau(series, MAX_LAG_TAU)
    else:
        tau = args.tau
    
    #埋め込み次元mを推定
    if args.m is None:
        E1 = itho_e1(series, max_dim=MAX_DIM, tau=tau)
        m = estimate_min_dimension(E1)
    else:
        m = args.m

    print(f"Using tau = {tau}, m = {m}, eps = {args.eps}")

    #埋め込み&リカレンス行列
    X = _embed(series, m, tau)
    if X.size == 0:
        raise RuntimeError("埋め込み後のデータが空です。データ長に対してτかmが大きい可能性があります")
    
    R = recurrence_matrix(X, eps=args.eps, normalize=True)

    #プロット
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(R, origin="lower", cmap="binary")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Time index")
    ax.set_title(f"Recurrence Plot (eps={args.eps}, tau={tau}, m={m})")
    plt.tight_layout()

    #画像保存
    if args.out is None:
        base = sanitize_filename(args.input)
        out_path = f"{base}_RP_eps{args.eps}.png"
    else:
        out_path = args.out
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[SAVE] Your Recurrence Plot: {out_path}")

if __name__ == "__main__":
    main()
