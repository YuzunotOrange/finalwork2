# 2025-12-23
#ローレンツアトラクタ、ロジスティックス写像、sin波、ホワイトノイズに関して指標を決めRQAしたもの
#追加で一次元ブラウン運動とローレンツアトラクタ、sine波、ロジスティックス写像それぞれにホワイトノイズを施した指標を追加
#指標それぞれに関する値を出力する
#時系列の長さとしてすべて2000に固定して処理している
#指標それぞれに関して20%、50%、70%、90%の破損を加えて線形補間を施している
#破損後の時系列も出力している
#ホワイトノイズを0~1の一様なものから最大振幅の5%を基準としてホワイトノイズを施すように変更している
#ホワイトノイズ、欠損それぞれ数回処理を行い、平均を計算
#伊藤法のE2の処理も追加済み


import numpy as np
import pandas as pd
import re, os
import shutil
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

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


def add_white_noise_relative(series, ratio=0.05):
     """
    系列の最大振幅 (max - min) の ratio 倍を標準偏差とする
    ガウス白色雑音 N(0, sigma^2) を加える
    """
     series = np.asarray(series, dtype=float)
     
     amplitude=np.max(series) - np.min(series)
     noise_std = ratio * amplitude

     noise = noise_std * np.random.randn(len(series))
     return series + noise

def generate_lorenz_noisy(n_steps=10000, dt=0.01,
                          sigma=10.0, rho=28.0, beta=8/3,
                          noise_ratio=0.05):
    x = generate_lorenz(n_steps=n_steps, dt=dt, sigma=sigma, rho=rho, beta=beta)
    return add_white_noise_relative(x, ratio=noise_ratio)

def generate_sin_noisy(n_steps=10000, freq=1.0, dt=0.01, noise_ratio=0.05):
    x = generate_sin(n_steps=n_steps, freq=freq, dt=dt)
    return add_white_noise_relative(x, ratio=noise_ratio)

def generate_logistic_noisy(n_steps=10000, r=4.0, x0=0.3, discard=1000, noise_ratio=0.05):
    x = generate_logistic(n_steps=n_steps, r=r, x0=x0, discard=discard)
    return add_white_noise_relative(x, ratio=noise_ratio)

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

def save_timeseries_missing_sets(series_grouped,
                                 out_dir="plots_timeseries_missing",
                                 max_points_for_plot=2000):
    """
    series_grouped: {
        base_name: {
            "0%":  {"interp": 1次元配列, "mask": bool配列},
            "20%": {"interp": 1次元配列, "mask": bool配列},
            ...
        },
        ...
    }

    ・各 base_name × 各欠損率ごとに 1枚のPNGを作成
    ・時系列は先頭 max_points_for_plot (=2000) ステップだけを表示
    ・実データは青点、線形補間された点は赤点＋赤線で表示
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for base_name, cond_dict in series_grouped.items():
        for label, d in cond_dict.items():
            s = np.asarray(d["interp"])
            mask = np.asarray(d["mask"], dtype=bool)

            # ★先頭 2000 ステップだけを見る
            n_total = len(s)
            n = min(max_points_for_plot, n_total)

            t_plot = np.arange(n)
            s_plot = s[:n]
            mask_plot = mask[:n]

            # 赤線は「補間された点だけ」をつなぐ
            s_interp_only = np.full_like(s_plot, np.nan, dtype=float)
            s_interp_only[mask_plot] = s_plot[mask_plot]

            plt.figure(figsize=(8, 4))

            # 線形補間部分（赤線＋赤点）
            plt.plot(t_plot, s_interp_only, "-",
                     linewidth=1.5, color="red", label="Interpolated (linear)")
            plt.scatter(t_plot[mask_plot], s_plot[mask_plot],
                        color="red", s=15)

            # 実データ（青点）
            plt.scatter(t_plot[~mask_plot], s_plot[~mask_plot],
                        color="blue", s=15, label="Actual data")

            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.title(f"{base_name} (missing {label})")
            plt.grid(True, ls="--", alpha=0.5)
            plt.legend()

            # ★x軸を 0〜2000 に固定（見た目を揃える）
            plt.xlim(0, max_points_for_plot)

            fname = f"{sanitize_filename(base_name)}_missing_{label.replace('%','pct')}.png"
            out_path = os.path.join(out_dir, fname)

            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"[Saved] {out_path}")


# ====== あなたが貼ってくれた関数群（必要なインポートだけ足してそのまま利用） ======
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

# ---------- Ito E2（追加） ----------
def _e_star(x, m, tau, k=None, theiler: int = 0):
    x = np.asarray(x, dtype=float)
    Xm = _embed(x, m, tau)
    M1 = Xm.shape[0]
    if M1 <= 0:
        return np.nan

    idx_last = np.arange(M1) + m * tau
    if np.any(idx_last >= len(x)):
        ok = np.where(idx_last < len(x))[0]
        if ok.size == 0:
            return np.nan
        Xm = Xm[: ok[-1] + 1]
        idx_last = idx_last[: ok[-1] + 1]
        M1 = Xm.shape[0]
        if M1 <= 1:
            return np.nan

    k_default = max(1, int(0.05 * M1))
    k_eff = max(1, min(k if k is not None else k_default, M1 - 1))
    over_k = min(M1 - 1, k_eff + 20)

    nn = NearestNeighbors(n_neighbors=over_k + 1, algorithm="auto", metric="euclidean")
    nn.fit(Xm)
    _, index = nn.kneighbors(Xm)

    diffs = []
    for i in range(M1):
        cand = index[i, 1:]
        if theiler > 0:
            cand = cand[np.abs(cand - i) > theiler]
        if cand.size == 0:
            continue

        use = min(k_eff, cand.size)
        jj = cand[:use]

        xi_last = x[idx_last[i]]
        xj_last = x[idx_last[jj]]
        diffs.append(np.mean(np.abs(xi_last - xj_last)))

    return float(np.mean(diffs)) if diffs else np.nan


def itho_e2(x, max_dim: int = 10, tau: int = 5, k=None, theiler: int = 0):
    x = np.asarray(x, dtype=float)
    Estar = np.array([_e_star(x, m, tau, k, theiler) for m in range(1, max_dim + 1)], dtype=float)

    E2 = []
    for m in range(1, max_dim):
        a, b = Estar[m - 1], Estar[m]
        if not np.isfinite(a) or a == 0 or not np.isfinite(b):
            E2.append(np.nan)
        else:
            E2.append(b / a)
    return np.array(E2, dtype=float)

def save_e2_plot(name, e2_values, savepath):
    import matplotlib.pyplot as plt
    e2_values = np.asarray(e2_values, dtype=float)

    m = np.arange(1, len(e2_values) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(m, e2_values, marker="o", linewidth=1.5, label="E2(m)")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="White noise ref (E2≈1)")

    plt.title(f"Ito E2 — {name}")
    plt.xlabel("Embedding dimension m")
    plt.ylabel("E2 = E*(m+1)/E*(m)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(savepath, dpi=200)
    plt.close()
    print(f"[Saved E2] {savepath}")

def extract_e2_from_result(res_dict):
    """
    res_dict から E2_1, E2_2, ... を昇順で取り出して list にして返す
    """
    e2_items = []
    for k, v in res_dict.items():
        m = re.fullmatch(r"E2_(\d+)", str(k))
        if m:
            e2_items.append((int(m.group(1)), v))
    e2_items.sort(key=lambda x: x[0])
    return [float(v) if (v is not None and np.isfinite(v)) else np.nan for _, v in e2_items]


def save_e2_overlay_plot(curves, title, savepath):
    """
    curves: dict[str, list[float]]  e.g. {"Sine + white noise": [...], "White noise": [...], ...}
    """
    plt.figure(figsize=(7, 4))

    for label, e2_vals in curves.items():
        e2_vals = np.asarray(e2_vals, dtype=float)
        m = np.arange(1, len(e2_vals) + 1)
        plt.plot(m, e2_vals, marker="o", linewidth=1.5, label=label)

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="White noise ref (E2≈1)")
    plt.title(title)
    plt.xlabel("Embedding dimension m")
    plt.ylabel("E2 = E*(m+1)/E*(m)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
    print(f"[Saved E2 Overlay] {savepath}")




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


# ====== 解 析 ＋ CSV へ保存 ======

def analyze_series(name, series, max_dim=10, max_lag=100, target_rr=0.05,
                   max_points_for_rqa=2000):  # ★引数追加
    print(f"\n===== {name} =====")
    x = np.asarray(series, dtype=float)
    x = (x - x.mean()) / (x.std() if x.std() > 0 else 1)

    tau = determine_tau(x, max_lag=max_lag)
    print(f"tau = {tau}")

    min_m_fixed = 2
    max_m_fixed = 4

    E1 = itho_e1(x, max_dim=max_m_fixed, tau = tau)

    m_est_raw = estimate_min_dimension(E1, eps=0.50, win=3)
    m_est = max(min_m_fixed, min(m_est_raw, max_m_fixed))

    print(f"estimated m (raw) = {m_est_raw}")
    print(f"used m (clipped to [{min_m_fixed}, {max_m_fixed}]) = {m_est}")

    X_tmp = _embed(x, m_est, tau)

    # ★RQA に使う点数をサンプリングして減らす
    N = X_tmp.shape[0]
    if N > max_points_for_rqa:
        step = int(np.ceil(N / max_points_for_rqa))
        X_rqa = X_tmp[::step]
        print(f"Using {X_rqa.shape[0]} points out of {N} for RQA (step={step})")
    else:
        X_rqa = X_tmp

    # ★eps もサンプリング後のデータで計算
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
    

    # ===== ★ Ito E2 を追加 =====
    # E2は m=1..(max_dim-1) で出る（長さ max_dim-1）
    # ここでは解析の重さを抑えるため、E1と同じく max_m_fixed を使う（必要なら max_dim に変更OK）
    E2 = itho_e2(x, max_dim=max_dim, tau=tau, k=None, theiler=0)

    # CSVに入れやすいように、E2_1, E2_2... の列に展開
    for i, v in enumerate(E2, start=1):
        result[f"E2_{i}"] = float(v) if np.isfinite(v) else np.nan

    # 要約値（便利）
    result["E2_mean"] = float(np.nanmean(E2)) 
    result["E2_last"] = float(E2[-1]) if np.isfinite(E2[-1]) else np.nan
    result["E2_values"] = E2.tolist()

    return result


def analyze_with_noise_ensemble(
    series_name,
    base_series,
    missing_rate,
    block_len,
    missing_seed,
    n_trials=10,
    noise_ratio=0.05,
    max_dim=10,
    max_lag=200,
    target_rr=0.05,
    vary_missing_each_trial=False,  # Trueなら欠損位置も毎回変える
):
    trial_results = []

    for t in range(n_trials):
        # ノイズは毎回変える（再現性が欲しければ seed を固定してずらす）
        np.random.seed(missing_seed + t)

        s_noisy = add_white_noise_relative(base_series, ratio=noise_ratio)

        seed_for_missing = (missing_seed + t) if vary_missing_each_trial else missing_seed
        s_proc, mask = introduce_block_missing_and_interpolate(
            s_noisy,
            missing_rate=missing_rate,
            block_len=block_len,
            seed=seed_for_missing
        )

        res = analyze_series(
            f"{series_name} (trial {t})",
            s_proc,
            max_dim=max_dim,
            max_lag=max_lag,
            target_rr=target_rr
        )
        trial_results.append(res)

    df = pd.DataFrame(trial_results)

    # 数値列だけ平均（文字列列は1つ目を採用）
    mean_res = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_res[col] = float(df[col].mean())
        else:
            mean_res[col] = df[col].iloc[0]

    # 参考：標準偏差も入れたい場合
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_res[col + "_std"] = float(df[col].std(ddof=1))

    return mean_res


def main():
    n = 2000
    np.random.seed(0)

    ROOT_OUT_DIR = "output_missing_analysis"
    if os.path.exists(ROOT_OUT_DIR):
        shutil.rmtree(ROOT_OUT_DIR)
    os.makedirs(ROOT_OUT_DIR, exist_ok=True)

    # --- ノイズ無しの元系列（5系） ---
    base_series = {
        "Lorenz attractor (x)": generate_lorenz(n_steps=n),
        "Sine wave":            generate_sin(n_steps=n),
        "Logistic map":         generate_logistic(n_steps=n),
        "White noise":          generate_white_noise(n_steps=n),
        "Brownian motion":      generate_brownian(n_steps=n),
    }

    # --- 解析したい “8システム” を定義 ---
    # name: 表示名
    # base_key: base_series のキー
    # noisy: ノイズ付けるか
    # ensemble: noisy のときだけ平均を取るか
    systems = [
        {"name": "Lorenz attractor (x)",              "base_key": "Lorenz attractor (x)", "noisy": False, "ensemble": False},
        {"name": "Logistic map",                      "base_key": "Logistic map",         "noisy": False, "ensemble": False},
        {"name": "Sine wave",                         "base_key": "Sine wave",            "noisy": False, "ensemble": False},
        {"name": "White noise",                       "base_key": "White noise",          "noisy": False, "ensemble": False},
        {"name": "Brownian motion",                   "base_key": "Brownian motion",      "noisy": False, "ensemble": False},

        {"name": "Lorenz + white noise",              "base_key": "Lorenz attractor (x)", "noisy": True,  "ensemble": True},
        {"name": "Logistic + white noise",            "base_key": "Logistic map",         "noisy": True,  "ensemble": True},
        {"name": "Sine + white noise",                "base_key": "Sine wave",            "noisy": True,  "ensemble": True},
    ]

    # ノイズ強度
    NOISE_RATIO = 0.05
    N_TRIALS = 5

    missing_rates = [0.0, 0.2, 0.5, 0.7, 0.9]
    missing_block_len = 10
    missing_seed = 0

    all_results = []
    series_grouped = {sys["name"]: {} for sys in systems}

    # E2プロット保存先
    e2_dir = os.path.join(ROOT_OUT_DIR, "E2_plots")
    os.makedirs(e2_dir, exist_ok=True)

    for sys in systems:
        base = base_series[sys["base_key"]]

        for r in missing_rates:
            cond_label = f"{int(r*100)}%"
            series_name = f'{sys["name"]} (missing {cond_label})'

            if sys["ensemble"]:
                # --- ノイズ付き3系：複数回まわして指標を平均 ---
                trial_results = []
                rep_s_proc = None
                rep_mask = None

                for t in range(N_TRIALS):
                    np.random.seed(missing_seed + t)  # ノイズ再現性

                    s_noisy = add_white_noise_relative(base, ratio=NOISE_RATIO)

                    # 欠損位置を固定したいなら seed は missing_seed 固定
                    s_proc, mask = introduce_block_missing_and_interpolate(
                        s_noisy, missing_rate=r, block_len=missing_block_len, seed=missing_seed
                    )

                    if t == 0:
                        rep_s_proc, rep_mask = s_proc, mask

                    res_t = analyze_series(
                        f"{series_name} (trial {t})",
                        s_proc,
                        max_dim=10, max_lag=200, target_rr=0.05
                    )
                    trial_results.append(res_t)

                df_trials = pd.DataFrame(trial_results)

                # 数値列だけ平均（文字列は先頭）
                mean_res = {}
                for col in df_trials.columns:
                    if pd.api.types.is_numeric_dtype(df_trials[col]):
                        mean_res[col] = float(df_trials[col].mean())
                    else:
                        mean_res[col] = df_trials[col].iloc[0]

                # メタ情報
                mean_res["name"] = series_name + f" + noise(avg{N_TRIALS})"
                mean_res["base_name"] = sys["name"]
                mean_res["missing_rate"] = r
                mean_res["block_len"] = missing_block_len
                mean_res["n_trials"] = N_TRIALS
                mean_res["noise_ratio"] = NOISE_RATIO

                all_results.append(mean_res)

                # 欠損系列PNG（代表trial0）
                series_grouped[sys["name"]][cond_label] = {"interp": rep_s_proc, "mask": rep_mask}

                # E2プロット（平均した E2_1, E2_2,... を使う）
                e2_items = []
                for k, v in mean_res.items():
                    m = re.fullmatch(r"E2_(\d+)", k)
                    if m:
                        e2_items.append((int(m.group(1)), v))
                e2_items.sort(key=lambda x: x[0])
                e2_vals = [float(v) if v is not None else np.nan for _, v in e2_items]

                save_name = f'{sanitize_filename(sys["name"])}_missing_{int(r*100)}pct_E2.png'
                save_path = os.path.join(e2_dir, save_name)
                save_e2_plot(f'{sys["name"]} (missing {int(r*100)}%) [avg{N_TRIALS}]', e2_vals, save_path)

            else:
                # --- ノイズ無し5系：1回だけ ---
                s_proc, mask = introduce_block_missing_and_interpolate(
                    base, missing_rate=r, block_len=missing_block_len, seed=missing_seed
                )

                res = analyze_series(series_name, s_proc, max_dim=10, max_lag=200, target_rr=0.05)

                res["base_name"] = sys["name"]
                res["missing_rate"] = r
                res["block_len"] = missing_block_len
                all_results.append(res)

                series_grouped[sys["name"]][cond_label] = {"interp": s_proc, "mask": mask}

                # E2プロット（単発）
                e2_items = []
                for k, v in res.items():
                    m = re.fullmatch(r"E2_(\d+)", k)
                    if m:
                        e2_items.append((int(m.group(1)), v))
                e2_items.sort(key=lambda x: x[0])
                e2_vals = [float(v) if v is not None else np.nan for _, v in e2_items]

                save_name = f'{sanitize_filename(sys["name"])}_missing_{int(r*100)}pct_E2.png'
                save_path = os.path.join(e2_dir, save_name)
                save_e2_plot(f'{sys["name"]} (missing {int(r*100)}%)', e2_vals, save_path)

    # CSV保存
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(ROOT_OUT_DIR, "results_missing_compare.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved: {csv_path}")
    
        # ===== E2 重ね描き（0%だけ：Sine+WN / White noise / Logistic+WN） =====
    overlay_dir = os.path.join(ROOT_OUT_DIR, "E2_overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    overlay_targets = [
        "Sine + white noise",
        "White noise",
        "Logistic + white noise",
    ]

    r = 0.0  # 0%だけ

    curves = {}
    for base_name in overlay_targets:
        cand = None
        for res in all_results:
            if res.get("base_name") == base_name and float(res.get("missing_rate", -1)) == float(r):
                cand = res
                break
        if cand is None:
            continue
        curves[base_name] = extract_e2_from_result(cand)

    if len(curves) >= 2:
        save_path = os.path.join(overlay_dir, "E2_overlay_missing_0pct.png")
        save_e2_overlay_plot(curves, "E2 Overlay (missing 0%)", save_path)
    else:
        print(f"[Skip overlay] missing 0% : not enough curves found -> {list(curves.keys())}")


    # 欠損系列PNG保存（8系すべて）
    plots_dir = os.path.join(ROOT_OUT_DIR, "plots_timeseries_missing")
    save_timeseries_missing_sets(series_grouped, out_dir=plots_dir, max_points_for_plot=2000)
    print(f"All plots saved under: {plots_dir}")



if __name__ == "__main__":
    main()

