# 2025-11-13
# rqa_experiment_new_grouping.py
# 実行方法: python rqa_experiment_new_grouping.py
# new_li/ 内の全CSVを自動解析し、rqa_result/ にRQA結果を保存（TT含む完全指標）
# DET は別グラフとして保存
#連続性(Laminatory)とTTに関する2Dグラフも追加で保存
#解析結果のグルーピングも実装
#グルーピングの方法としてk-meansを使用することにより自動的に閾値を指定しグルーピングしている
#ソートの方法として決定性の値に着目し実装していたがDET/RRのDEratioの値に着目するように仕様を変更している


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
# --- RQAパラメータ空間の散布図（3D & 2D） ---
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 20,   # 軸ラベルサイズ
    "axes.titlesize": 22,   # タイトル
})



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

# ---- 追加: TT を ε ごとに計算する補助関数 ----
def trapping_time_curve(X, epsilons, vmin=2):
    TTs = []
    for e in epsilons:
        R = recurrence_matrix(X, e)
        vls = vertical_line_lengths(R)
        vsel = vls[vls >= vmin]
        TTs.append(np.mean(vsel) if len(vsel)>0 else 0.0)
    return np.array(TTs, dtype=float)

# ---- グルーピング関数 ----
def assign_group(df, key="mD", method="tertile", thresholds=None, labels=None):
    """
    method:
        'tertile' : 3分位　（上位/中位/下位）
        'quartile' : 四分位（Q4/Q3/Q2/Q1）
        'threshold': thresholds で任意境界（高→低の順に判定）
    """
    x = df[key].astype(float)
    out = df.copy()

    if method == "threshold":
        th = thresholds or []
        if labels is None:
            labels = ["High", "Mid", "Low", "VeryLow"][:len(th)+1]
        #高い方から順に割り当てていく
        group_vals =[]
        for val in x:
            assigned = False
            for i, t in enumerate(th):
                if val >= t:
                    group_vals.append(labels[i]); assigned=True; break
            if not assigned:
                group_vals.append(labels[len(th)])
        out["group"] = group_vals
        return out
    
    if method == "quartile":
        #低->高の順番のラベルになるため反転して上位を高ランクに並び変える
        q_labels = ["Q1(Low)", "Q2", "Q3", "Q4(High)"]
        out["group"] = pd.qcut(x, q=4, labels=q_labels, duplicates="drop")
        return out
    
    #----- k-meansのクラスタリングの実装-----
    if method == "kmeans":
        #1次元ベクトルに変形
        X = x.to_numpy().reshape(-1, 1)

        #k-meansでクラスタ
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        labels_raw = km.fit_predict(X) #0,1,2のクラスタID

        #各クラスタの平均値を求めて「高い順」に変更する
        cluster_means = (
            pd.Series(x.values, index=labels_raw)
            .groupby(level=0)
            .mean()
            .sort_values(ascending=False)
        )

        order = cluster_means.index.to_list()
        name_list = ["High", "Mid", "Low"]
        cluster_to_name = {c: name_list[i] for i, c in enumerate(order)}

        out["group"] = [cluster_to_name[c] for c in labels_raw]
        return out 
    
    "default: tertile"
    t_labels = ["Low", "Mid", "High"]
    out["group"] = pd.qcut(x, q=3, labels=t_labels, duplicates="drop")
    return out

# ---- ソート棒グラフ ----
def plot_sorted_bar(df, key="mD", path="rqa_result/sorted_bar.png"):
    plt.figure(figsize=(16, 6))
    
    #ラベル
    labels =df["file"].astype(str) if "file" in df.columns else df.index.astype(str)
    
    if "group" in df.columns:
        groups = df["group"].astype("str")

        # 共通のグループカラーを使用
        def color_for_group(g):
            return GROUP_COLORS.get(g, plt.cm.tab20(3))
        
        colors = groups.map(color_for_group)

        plt.bar(range(len(df)), df[key].astype(float).values, color=colors)
        unique_groups = groups.unique()

        #グループごとに凡例を出すため、散布でダミーを作成
        handles = [plt.Rectangle((0,0), 1, 1, color=color_for_group(g))
                   for g in unique_groups]
        plt.legend(handles, unique_groups, title="Group", loc="best")
    else:
        plt.bar(range(len(df)), df[key].astype(float).values)

    step = max(1, len(df) // 60)
    idx = range(0, len(df), step)  
    plt.xticks(idx, labels.iloc[idx], rotation=75, ha="right")
    plt.ylabel(key)
    plt.title(f"Sorted by {key} (descending)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_rqa_param_space(df, out_dir=MAIN_OUT_DIR,
                         path3d="rqa_param_space_3d.png",
                         path2d="rqa_param_space_pairs.png"):
    """
    df: 列 'mD','mL','mEN'（必須）、'file','group','rank'（任意）
    - 3D: (mD, mL, mEN) の散布図
    - 2D: (mD vs mL), (mD vs mEN), (mL vs mEN) の3枚
    - 色分け: df['group'] があればそれ（Low/Mid/High 等）、なければ単色
    """
    need = {"mDEratio", "mL", "mEN"}
    if not need <= set(df.columns):
        missing = need - set(df.columns)
        raise SystemExit(f"[ERROR] plot_rqa_param_space: 必須列が不足しています: {missing}")

    d = df.copy()
    # グループ列が無ければダミーを作る
    if "group" not in d.columns:
        d["group"] = "All"

    # 色マップ（カテゴリ→色）
    groups = d["group"].astype("category")
    
    def color_for_group(g):
        return GROUP_COLORS.get(str(g), plt.cm.tab20(3))
    
    colors = [color_for_group(g) for g in groups]

    # ===== 3D 散布図 =====
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(d["mDEratio"].astype(float), d["mL"].astype(float), d["mEN"].astype(float),
                    c=colors, s=40, depthshade=True)

    ax.set_xlabel("mDEratio (mean DET)")
    ax.set_ylabel("mL (mean L)")
    ax.set_zlabel("mEN (mean ENTR)")
    ax.set_title("RQA Parameter Space (mDEratio, mL, mEN)")

    # 凡例（グループごと）
    handles = [plt.Line2D([0],[0], marker="o", linestyle="",
                                  color=color_for_group(g), label=str(g))
                for g in groups.cat.categories]
        
    ax.legend(handles=handles, title="Group", loc="best")

    plt.tight_layout()
    f3d = os.path.join(out_dir, path3d)
    fig.savefig(f3d, dpi=200, bbox_inches="tight")
    plt.close(fig)

        # ===== 2D ペア散布図 =====
    base_pairs = [("mDEratio","mL"), ("mDEratio","mEN"), ("mL","mEN")]
    out_full = os.path.join(out_dir, path2d)

    # --- 拡大なし（全体）
    scatter_pairs(d, base_pairs, colors, groups,
                out_path=out_full,
                title="RQA Parameter Space (2D pairs)")

    # --- 拡大あり（指定レンジ）
    zoom_ranges = {
        ("mDEratio","mL"):  {"xlim": (5, 24), "ylim": (0, 50)},
        ("mDEratio","mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},  # 必要に応じて (2,3) などへ
        ("mL","mEN"): {"xlim": (0, 20),   "ylim": (1, 3.5)},
    }
    f2d_zoom = os.path.join(out_dir, "rqa_param_space_pairs_zoom.png")
    scatter_pairs(d, base_pairs, colors, groups, 
                out_path=f2d_zoom,
                title="RQA Parameter Space (2D pairs, ZOOM)",
                zoom_ranges=zoom_ranges,
                annotate=True)

    # --- TT の図（別ファイル）
    if "mTT" in d.columns:
        tt_pairs = [("mDEratio","mTT"), ("mL","mTT"), ("mEN","mTT")]
        out_tt_full = os.path.join(out_dir, "rqa_param_space_pairs_TT.png")
        scatter_pairs(d, tt_pairs, colors, groups, 
                    out_path=out_tt_full,
                    title="RQA Parameter Space (TT pairs)")
        
        #---拡大図
        tt_zoom_range = {
            ("mDEratio", "mTT"): {"xlim": (10, 25), "ylim":(0, 45)},
            ("mL", "mTT"): {"xlim": (0, 25), "ylim": (0, 40)},
            ("mEN", "mTT"): {"xlim": (0, 3), "ylim":(0, 40)},
        }

        out_tt_zoom = os.path.join(out_dir, "rqa_param_space_pairs_TT_zoom.png")
        
        scatter_pairs(
            d, tt_pairs, colors, groups,
            out_path=out_tt_zoom,
            title="RQA Parameter Space (TT pairs, ZOOM)",
            zoom_ranges=tt_zoom_range,
            annotate=True
        )

        print(f"[Saved] {out_tt_full}")
        print(f"[Saved] {out_tt_zoom}")
    
    
    # --- TT の図（別ファイル）
    if "mLAM" in d.columns:
        lam_tt_pairs = [("mLAM","mTT")]
        out_lam_tt_full = os.path.join(out_dir, "rqa_param_space_pairs_LAM_TT.png")
        scatter_pairs(d, lam_tt_pairs, colors, groups, 
                    out_path=out_lam_tt_full,
                    title="RQA Parameter Space (LAM vs TT)")
        
        #---拡大図
        lam_tt_zoom_range = {
            ("mLAM", "mTT"): {"xlim": (0.6, 1.0), "ylim":(0, 50)},
        }
        
        out_lam_tt_zoom = os.path.join(out_dir, "rqa_param_space_pairs_LAM_TT_zoom.png")
                
        scatter_pairs(
            d, lam_tt_pairs, colors, groups, 
            out_path=out_lam_tt_zoom,
            title="RQA Parameter Space (LAM vs TT, ZOOM)",
            zoom_ranges=lam_tt_zoom_range,
            annotate=True
        )
        

        print(f"[Saved] {out_lam_tt_full}")
        print(f"[Saved] {out_lam_tt_zoom}")

    # === CSV保存はそのまま ===
    cols = ["file","group","rank","mD","mL","mEN","mTT", "LAM", "mDEratio"]
    cols = [c for c in cols if c in d.columns]
    d[cols].to_csv(os.path.join(out_dir, "rqa_param_space_points.csv"), index=False)

    print(f"[Saved] {f3d}")
    print(f"[Saved] {out_full}")
    print(f"[Saved] {f2d_zoom}")
    if "mTT" in d.columns:
        print(f"[Saved] {os.path.join(out_dir, 'rqa_param_space_pairs_TT.png')}")
        print(f"[Saved] {os.path.join(out_dir, 'rqa_param_space_pairs_TT_zoom.png')}")
    print(f"[Saved] {out_dir}/rqa_param_space_points.csv")


# ====== メイン処理 ======
def process_csv(path, eps_min, eps_max, eps_steps, lmin):
    print(f"\n=== Processing: {path} ===")
    series = load_series_first_col_auto(path)
    tau = 5 if os.path.basename(path).lower()=="lorenz.csv" else determine_tau(series)
    k = int(max(1, K_FRAC*len(series)))
    E1 = itho_e1(series, tau=tau, k=k, theiler=int(tau))
    m = min(estimate_min_dimension(E1), MAX_DIM)
    epsilons = np.linspace(eps_min, eps_max, eps_steps)
    dets,Ls,ENs,avg,X = compute_rqa_from_tau_m(series,tau,m,epsilons,lmin)

    base = sanitize_filename(path)
    out_dir = os.path.join(MAIN_OUT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    #-----TTに関するグルーピングのグラフを追加-----
    mTT = 0.0
    TTs = np.array([])
    if X.size > 0:
        TTs = trapping_time_curve(X, epsilons, vmin=lmin)
        if TTs.size > 0:
            pos = TTs[TTs > 0]
            mTT = float(np.mean(pos)) if pos.size > 0 else 0.0

    # --- DET以外 (L, EN) のRQAカーブ ---
    if dets:
        plt.figure(figsize=(10,4))
        plt.plot(epsilons,Ls,label="Mean Line Length (L)")
        plt.plot(epsilons,ENs,label="Entropy (EN)")
        plt.title(f"RQA (L, EN) vs ε (τ={tau}, m={m})")
        plt.xlabel("ε"); plt.legend(); plt.tight_layout()
        plt.yscale("log")
        plt.savefig(os.path.join(out_dir,f"rqa_curves_{base}.png"),dpi=200)
        plt.close()

    # --- DET のみ別グラフ ---
    if dets:
        plt.figure(figsize=(8,4))
        plt.plot(epsilons,dets,color="C3",label="Determinism (DET)")
        plt.title(f"Determinism vs ε (τ={tau}, m={m})")
        plt.xlabel("ε"); plt.ylabel("DET")
        plt.grid(True,ls="--",alpha=0.6)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"determinism_curve_{base}.png"),dpi=200)
        plt.close()

        # --- 追加: TT vs ε と L vs ε を重ね描き（左右軸） & CSV保存 ---
    if X.size>0 and len(Ls)>0:
        TTs = trapping_time_curve(X, epsilons, vmin=lmin)
        # CSV 保存
        pd.DataFrame({
            "epsilon": epsilons,
            "TT": TTs if TTs.size>0 else np.zeros_like(epsilons),
            "L": Ls
        }).to_csv(os.path.join(out_dir, f"tt_l_curves_{base}.csv"), index=False)

        # 図（重ね描き：左軸TT、右軸L）
        fig, ax1 = plt.subplots(figsize=(10,4))
        ax2 = ax1.twinx()
        line1, = ax1.plot(epsilons, TTs, label="Trapping Time (TT)")
        line2, = ax2.plot(epsilons, Ls, linestyle="--", label="Mean Line Length (L)")
        ax1.set_xlabel("ε")
        ax1.set_ylabel("TT")
        ax2.set_ylabel("Mean Line Length (L)")
        ax1.grid(True, ls="--", alpha=0.4)
        plt.title(f"TT & Mean Line Length vs ε (τ={tau}, m={m})")

        # 共同凡例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"tt_l_curve_{base}.png"), dpi=200)
        plt.close(fig)

    # --- RP + full measures (TT含む) ---
    mLAM = np.nan  # 連続性 LAM の平均値（RR=0.05 のRPから）
    mDEratio = np.nan #DEratio DET/RR　の指標
    if X.size > 0:
        eps_rp = epsilon_for_target_rr(X)
        R = recurrence_matrix(X, eps_rp)
        plt.figure(figsize=(5,5))
        plt.imshow(R, origin="lower", cmap="gray_r")
        plt.title(f"RP (ε≈{eps_rp:.3f}, RR={R.mean():.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rp_{base}.png"), dpi=200)
        plt.close()

        full = rqa_measures_full(R, lmin, lmin, lmin)
        pd.DataFrame([full]).to_csv(
            os.path.join(out_dir, f"full_measures_rr5_{base}.csv"),
            index=False
        )

        # full の中に LAM があれば取得
        if "LAM" in full:
            mLAM = float(full["LAM"])
        if "DEratio" in full:
            mDEratio = float(full["DEratio"])

    summary = {
        "file": os.path.basename(path),
        "tau": tau,
        "m": m,
        "mD": avg["mD"],
        "mL": avg["mL"],
        "mEN": avg["mEN"],
        "mTT": mTT,
        "mLAM": mLAM,
        "mDEratio": mDEratio,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, f"result_summary_{base}.csv"),
        index=False
    )

    return summary

def scatter_pairs(d, pairs, colors, groups, out_path,
                  title, zoom_ranges=None, figsize_scale=5, annotate=True):
    """
    pairs: [("x","y"), ...]
    zoom_ranges: {("x","y"): {"xlim":(a,b), "ylim":(c,d)}, ...} or None
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_scale*n, 4.5),
                            constrained_layout=True)
    if n == 1:
        axes = [axes]
    
    colors_arr = np.array(colors)

    for ax, (xcol, ycol) in zip(axes, pairs):
        x = d[xcol].astype(float)
        y = d[ycol].astype(float)
        
        # ラベル番号
        lbl = (d["rank"] if "rank" in d.columns else (d.index + 1)).values
        
        mask = np.ones_like(x, dtype=bool)
        if zoom_ranges and (xcol, ycol) in zoom_ranges:
            xr, yr = zoom_ranges[(xcol, ycol)]["xlim"], zoom_ranges[(xcol, ycol)]["ylim"]
            mask = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
            ax.set_xlim(*xr); ax.set_ylim(*yr)


        # 可視範囲のみ描画（←ここが肝）
        ax.scatter(x[mask], y[mask], c=colors_arr[mask], s=24, alpha=0.7, edgecolors="none")

        if annotate:
            for xi, yi, la in zip(x[mask], y[mask], lbl[mask]):
                ax.text(float(xi), float(yi), str(la), fontsize=7,
                        ha="center", va="bottom", alpha=0.7, clip_on=True) # ← clip_on

        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    # 共同凡例
    def color_for_group(g):
        return GROUP_COLORS.get(str(g), plt.cm.tab20(3))
    handles = [plt.Line2D([0],[0], marker="o", linestyle="",
                          color=color_for_group(g), label=str(g))
               for g in groups.cat.categories]
    axes[-1].legend(handles=handles, title="Group", loc="best")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_max", type=float, default=2.0)
    parser.add_argument("--eps_steps", type=int, default=20)
    parser.add_argument("--lmin", type=int, default=2)
    # ---- 追加　ソート&グループ設定 ----
    parser.add_argument("--sort_key", choices=["mD", "mL", "mEN", "mTT", "mDEratio"], default="mDEratio", help="集計ソート（降順)")
    parser.add_argument("--group_method", choices=["tertile", "quartile", "threshold", "kmeans"], default="tertile", help="グルーピング手法")
    parser.add_argument("--threshold", type=str, default="", help="group_method=threshold 用の境界（高->低）")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise SystemExit(f"[ERROR] 入力フォルダ {args.dir} が存在しません。")

    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        raise SystemExit(f"[ERROR] {args.dir} に CSV が見つかりません。")

    results=[]
    for f in files:
        try:
            results.append(process_csv(f,args.eps_min,args.eps_max,args.eps_steps,args.lmin))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
    
    # ---- 集計csv (保存の処理だけ)----
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(MAIN_OUT_DIR,"rqa_summary.csv"),index=False)

    # ---- ソート（降順）＆　ランク保持 ---- 
    sort_key = "mDEratio" #決定比でソート
    df_sorted = df.sort_values(by=sort_key, ascending=False).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted)+1)

    # ---- グルーピング ----
    thresholds = None
    if args.group_method == "threshold":
        if args.threshold.strip():
            try:
                thresholds = [float(s) for s in args.threshold.split(",")]
            except Exception:
                raise SystemExit("[ERROR] --thresholds　はカンマ区切りで数値を指定してください")
        
    df_grouped = assign_group(df_sorted, key="mDEratio", method=args.group_method, thresholds=thresholds)

    # ---- 追加保存　＆　可視化 ----
    out_sorted_csv = os.path.join(MAIN_OUT_DIR, "rqa_summary_sort.csv")
    df_grouped.to_csv(out_sorted_csv, index=False)
    print(f"\n[Saved] {out_sorted_csv}")

    plot_sorted_bar(df_grouped, key=sort_key, path=os.path.join(MAIN_OUT_DIR, f"sorted_bar_{sort_key}.png"))
    # RQAパラメータ空間の散布図（3D & 2D）
    plot_rqa_param_space(df_grouped, out_dir=MAIN_OUT_DIR,
                     path3d="rqa_param_space_3d.png",
                     path2d="rqa_param_space_pairs.png")

    print(f"\n[Saved] {MAIN_OUT_DIR}/sorted_bar_{sort_key}.png")
    

    #---- グループ別の一覧も出力 ----
    for g, gdf in df_grouped.groupby("group", sort=False):
        gpath = os.path.join(MAIN_OUT_DIR, f"goup_{g}.csv")
        gdf.to_csv(gpath, index=False)
        print(f"[Saved] {gpath}")


if __name__ == "__main__":
    main()
