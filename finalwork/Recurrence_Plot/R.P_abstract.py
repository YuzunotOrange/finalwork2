import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors

# =========================
# 設定
# =========================
XMIN = 0.0
XMAX = 600.0
DT   = 1.0  # 週刻み（必要なら変更）

OUT_DIR_NAME = "new_li_ts_rp"

# RQA設定
MAX_DIM = 10
K_FRAC  = 0.50
MAX_LAG_TAU = 100
TAU_MODE = "auto"   # "auto" or "manual"
TAU_VALUE = 5
M_MODE   = "auto"   # "auto" or "manual"
M_VALUE  = 3
TARGET_RR = 0.05
EPS_MANUAL = None   # eps固定したいなら数値、通常はNone


# =========================
# ユーティリティ
# =========================
def sanitize_filename(name):
    return re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(name))[0])

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / sd if sd > 0 else (x - mu)

def robust_load_csv_1or2cols(path):
    """
    1列: yのみ (timeは0..N-1を週として作る)
    2列以上: 先頭2列を (x,y) として読む
    ヘッダ/文字列が混じっても数値だけ拾う
    """
    df = pd.read_csv(path, header=None, comment="#", on_bad_lines="skip")
    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    df = df.dropna(axis=0, how="any")  # 行にNaNがあるのは落とす（厳しめ）

    if df.shape[1] >= 2:
        x = df.iloc[:, 0].to_numpy(float)
        y = df.iloc[:, 1].to_numpy(float)
        return x, y, "xy"
    elif df.shape[1] == 1:
        y = df.iloc[:, 0].to_numpy(float)
        x = np.arange(len(y), dtype=float)  # 0,1,2,... (週とみなす)
        return x, y, "yonly"
    else:
        raise ValueError("CSVに数値データが見つかりませんでした")

# =========================
# RQA関数（必要最小限）
# =========================
def calculate_autocorrelation(series, max_lag):
    s = np.asarray(series, dtype=float)
    mean = np.mean(s)
    denom = np.sum((s - mean) ** 2)
    vals = []
    for lag in range(1, max_lag + 1):
        if lag >= len(s):
            break
        num = np.sum((s[:-lag] - mean) * (s[lag:] - mean))
        vals.append(num / denom if denom != 0 else 0.0)
    return vals

def determine_tau(series, max_lag=100):
    ac = calculate_autocorrelation(series, max_lag)
    for i, v in enumerate(ac):
        if v < 1 / np.e:
            return i + 1
    return max_lag

def _embed(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m))
    idx = np.arange(N)[:, None] + tau * np.arange(m)[None, :]
    return x[idx]

def itho_e1(x, max_dim=10, tau=5, s=None, k=None, theiler=0):
    x = np.asarray(x, dtype=float)
    if s is None:
        s = tau
    if k is None:
        k = max(1, int(0.05 * len(x)))
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
        nn = NearestNeighbors(n_neighbors=min(k_eff + 21, valid_M)).fit(X_now)
        dists, idxs = nn.kneighbors(X_now)

        ratios = []
        for i in range(valid_M):
            j = idxs[i, 1:]
            d = dists[i, 1:]

            if theiler > 0:
                mask = np.abs(j - i) > theiler
                j, d = j[mask], d[mask]
            if j.size == 0:
                continue

            mask2 = (j + s) < M
            j, d = j[mask2], d[mask2]
            if j.size == 0:
                continue

            use = min(k_eff, j.size)
            jj = j[:use]
            dn = np.mean(d[:use])
            if dn < 1e-12:
                continue

            df = np.mean(np.linalg.norm(Xm[i + s] - Xm[jj + s], axis=1))
            ratios.append(df / dn)

        if ratios:
            E1.append(np.mean(ratios))

    return np.array(E1)

def estimate_min_dimension(E1, eps=0.05, win=3):
    E1 = np.asarray(E1)
    if E1.size == 0:
        return 1
    for m in range(1, len(E1) - win + 1):
        seg = E1[m : m + win]
        if np.all(np.abs(np.diff(seg)) / np.maximum(np.abs(seg[:-1]), 1e-12) <= eps):
            return m + 1
    return len(E1)

def recurrence_matrix(X, eps, normalize=True):
    if normalize:
        std = np.std(X)
        X = X / std if std > 0 else X
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    R = (D <= eps).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

def epsilon_for_target_rr(X, target_rr=0.05):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    flat = D[np.isfinite(D)]
    k = max(1, int(target_rr * len(flat)) - 1)
    return float(np.partition(flat, k)[k])

# =========================
# プロット（上下でx対応、タイトル/ラベル明記）
# =========================
def plot_ts_and_rp_stacked_week_axes(
    title_prefix,
    t_raw, y_raw,
    t_grid, y_grid,
    R, eps,
    out_png,
    xmin, xmax
):
    fig, (ax_ts, ax_rp) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios":[1.0, 1.2], "hspace":0.28},
        sharex=False
    )

    # ---- 上：時系列 ----
    ax_ts.plot(t_grid, y_grid, "-o", color="red", markersize=3.5, linewidth=1.2, label="Interpolated (linear)")
    ax_ts.scatter(t_raw, y_raw, s=28, color="blue", label="Actual data", zorder=3)

    ax_ts.set_xlim(xmin, xmax)
    ax_ts.set_title(f"{title_prefix}  Time series (weeks {int(xmin)}–{int(xmax)})")
    ax_ts.set_xlabel("Time [weeks]")
    ax_ts.set_ylabel("Flux")
    ax_ts.grid(True, alpha=0.35)
    ax_ts.legend(loc="best")

    # ★ 時系列が潰れて見えない対策：範囲内データからylimを作る
    y_all = np.concatenate([y_grid, y_raw]) if len(y_raw) else y_grid
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        pad = 0.08 * (y_max - y_min)
        ax_ts.set_ylim(y_min - pad, y_max + pad)

    # ---- 下：RP（軸を週に対応させる）----
    rr = float(R.mean())
    ax_rp.imshow(
        R,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=[xmin, xmax, xmin, xmax]   # ★ i,j を週に変換して表示
    )
    ax_rp.set_title(f"RP (ε≈{eps:.3f}, RR={rr:.3f})")
    ax_rp.set_xlabel("Time [weeks] (i)")
    ax_rp.set_ylabel("Time [weeks] (j)")
    ax_rp.set_xlim(xmin, xmax)
    ax_rp.set_ylim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# =========================
# メイン（一括処理）
# =========================
def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(target_dir, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in directory.")
        return

    summary = []

    for fname in csv_files:
        print(f"Processing: {fname}")
        base = sanitize_filename(fname)

        # --- 読み込み（1列/2列対応） ---
        x_data, y_data, mode = robust_load_csv_1or2cols(fname)

        # --- 週軸にする（あなたの元コード踏襲）
        # 2列CSVの場合：xが日なら週に変換したい → (x-x0)/7
        # 1列CSVの場合：xは0..N-1を週として作っているので変換しない
        if mode == "xy":
            x_week = (x_data - x_data[0]) / 7.0
        else:
            x_week = x_data  # すでに週扱い

        # --- 0–600 で raw を切る（青点）---
        mask_raw = (x_week >= XMIN) & (x_week <= XMAX)
        t_raw = x_week[mask_raw]
        y_raw = y_data[mask_raw]

        # rawが少なすぎる場合でも、補間系列は作れることがあるので一応続行
        # ただし補間は最低2点以上必要
        if len(x_week) < 2:
            print("  [Skip] Not enough data points for interpolation.")
            continue

        # --- 補間系列（0–600をDT刻み）---
        t_grid = np.arange(XMIN, XMAX + 1e-12, DT, dtype=float)

        # 外挿は許可（範囲内に観測が少ないときに線が作れない問題を避ける）
        f_curve = interpolate.interp1d(
            x_week, y_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        y_grid = f_curve(t_grid)

        # --- RPも 0–600 の補間系列のみ ---
        series = zscore(y_grid)

        # tau
        if TAU_MODE == "auto":
            tau = determine_tau(series, MAX_LAG_TAU)
        else:
            tau = int(TAU_VALUE)

        # m
        if M_MODE == "auto":
            k = int(max(1, K_FRAC * len(series)))
            E1 = itho_e1(series, tau=tau, k=k, theiler=int(tau))
            m = min(estimate_min_dimension(E1), MAX_DIM)
        else:
            m = int(M_VALUE)

        X = _embed(series, m, tau)
        if X.size == 0:
            print(f"  [Skip] Embedding failed (m={m}, tau={tau}).")
            continue

        # eps
        if EPS_MANUAL is not None:
            eps = float(EPS_MANUAL)
        else:
            eps = epsilon_for_target_rr(X, target_rr=TARGET_RR)

        R = recurrence_matrix(X, eps)

        # --- 出力（上：時系列、下：RP）---
        out_png = os.path.join(out_dir, f"{base}_ts_rp_0_600.png")
        plot_ts_and_rp_stacked_week_axes(
            title_prefix=base,
            t_raw=t_raw, y_raw=y_raw,
            t_grid=t_grid, y_grid=y_grid,
            R=R, eps=eps,
            out_png=out_png,
            xmin=XMIN, xmax=XMAX
        )
        print(f"  -> Saved: {out_png}")

        summary.append({
            "file": os.path.basename(fname),
            "mode": mode,
            "xmin": XMIN, "xmax": XMAX, "dt": DT,
            "n_raw_in_range": int(len(t_raw)),
            "tau": int(tau), "m": int(m),
            "eps": float(eps), "RR": float(R.mean()),
            "out_png": os.path.basename(out_png),
        })

    if summary:
        summary_path = os.path.join(out_dir, "summary_info.csv")
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"[Saved] {summary_path}")

    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()
