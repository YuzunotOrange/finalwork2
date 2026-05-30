import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

mpl.rcParams.update({
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
})

MAX_DIM = 10
K_FRAC  = 0.50
MAX_LAG_TAU = 100

DEFAULT_INPUT_DIR = "new_li"
MAIN_OUT_DIR = "rqa_result_LAM_TT"
DEFAULT_EXCLUDE_RANKS = [139, 11, 24]

GROUP_COLORS = {
    "Low":  plt.cm.tab20(0),
    "Mid":  plt.cm.tab20(1),
    "High": plt.cm.tab20(2),
}

# ===== 基本処理 =====
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

def determine_tau(series, max_lag=MAX_LAG_TAU):
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

def itho_e1(x, max_dim=MAX_DIM, tau=5, s=None, k=None, theiler=0):
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
        seg = E1[m:m + win]
        if np.all(np.abs(np.diff(seg)) / np.maximum(np.abs(seg[:-1]), 1e-12) <= eps):
            return m + 1
    return len(E1)

def load_series_first_col_auto(path):
    try:
        df = pd.read_csv(path, header=None, usecols=[0], comment="#", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, header=None, sep=None, engine="python", comment="#", usecols=[0])

    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").interpolate().bfill().ffill()
    x = s.to_numpy(dtype=float)
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / sd if sd > 0 else (x - mu)

def recurrence_matrix(X, eps, normalize=True):
    if normalize:
        sd = np.std(X)
        X = X / sd if sd > 0 else X
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    R = (D <= eps).astype(np.uint8)
    np.fill_diagonal(R, 0)
    return R

def vertical_line_lengths(R):
    N = len(R)
    Ls = []
    for j in range(N):
        col = R[:, j]
        run = 0
        for v in col:
            if v == 1:
                run += 1
            elif run > 0:
                Ls.append(run); run = 0
        if run > 0:
            Ls.append(run)
    return np.array(Ls, int)

def diag_line_lengths(R):
    N = len(R)
    Ls = []
    for k in range(-N + 1, N):
        diag = np.diag(R, k)
        run = 0
        for v in diag:
            if v == 1:
                run += 1
            elif run > 0:
                Ls.append(run); run = 0
        if run > 0:
            Ls.append(run)
    return np.array(Ls, int)

def epsilon_for_target_rr(X, target_rr=0.05):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    flat = D[np.isfinite(D)]
    k = max(1, int(target_rr * len(flat)) - 1)
    return float(np.partition(flat, k)[k])

def rqa_measures_full_core(R, lmin=2, vmin=2):
    rr = float(R.mean())

    dls = diag_line_lengths(R)
    dsel = dls[dls >= lmin]
    det = np.sum(dsel) / R.sum() if len(dsel) > 0 and R.sum() > 0 else 0.0
    deratio = det / rr if rr > 0 else np.nan

    vls = vertical_line_lengths(R)
    vsel = vls[vls >= vmin]
    lam = np.sum(vsel) / R.sum() if len(vsel) > 0 and R.sum() > 0 else 0.0
    tt = np.mean(vsel) if len(vsel) > 0 else 0.0

    return {"LAM": lam, "TT": tt, "DEratio": deratio}

def process_csv_for_metrics(path, vmin=2):
    series = load_series_first_col_auto(path)
    tau = determine_tau(series)

    k = int(max(1, K_FRAC * len(series)))
    E1 = itho_e1(series, tau=tau, k=k, theiler=int(tau))
    m = min(estimate_min_dimension(E1), MAX_DIM)

    X = _embed(series, m, tau)
    if X.size == 0:
        return {"file": os.path.basename(path), "tau": tau, "m": m,
                "mLAM": np.nan, "mTT": np.nan, "mDEratio": np.nan}

    eps_rp = epsilon_for_target_rr(X, target_rr=0.05)
    R = recurrence_matrix(X, eps_rp)
    full = rqa_measures_full_core(R, lmin=vmin, vmin=vmin)

    return {
        "file": os.path.basename(path),
        "tau": tau,
        "m": m,
        "mLAM": float(full["LAM"]),
        "mTT": float(full["TT"]),
        "mDEratio": float(full["DEratio"]),
    }

# ===== グルーピング / 棒グラフ =====
def assign_group(df, key="mDEratio", method="tertile", thresholds=None, labels=None):
    x = df[key].astype(float)
    out = df.copy()

    if method == "threshold":
        th = thresholds or []
        if labels is None:
            labels = ["High", "Mid", "Low", "VeryLow"][:len(th) + 1]
        group_vals = []
        for val in x:
            assigned = False
            for i, t in enumerate(th):
                if val >= t:
                    group_vals.append(labels[i]); assigned = True; break
            if not assigned:
                group_vals.append(labels[len(th)])
        out["group"] = group_vals
        return out

    if method == "quartile":
        q_labels = ["Q1(Low)", "Q2", "Q3", "Q4(High)"]
        out["group"] = pd.qcut(x, q=4, labels=q_labels, duplicates="drop")
        return out

    if method == "kmeans":
        X = x.to_numpy().reshape(-1, 1)
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        labels_raw = km.fit_predict(X)

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

    t_labels = ["Low", "Mid", "High"]
    out["group"] = pd.qcut(x, q=3, labels=t_labels, duplicates="drop")
    return out

def plot_sorted_bar(df, key="mDEratio", path="sorted_bar.png"):
    plt.figure(figsize=(16, 6))
    labels = df["file"].astype(str) if "file" in df.columns else df.index.astype(str)

    if "group" in df.columns:
        groups = df["group"].astype("str")

        def color_for_group(g):
            return GROUP_COLORS.get(g, plt.cm.tab20(3))

        colors = groups.map(color_for_group)
        plt.bar(range(len(df)), df[key].astype(float).values, color=colors)

        unique_groups = groups.unique()
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_for_group(g)) for g in unique_groups]
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

# ===== LAM vs TT 図 =====
def scatter_lam_tt(df, out_path, title, xlim=None, ylim=None, annotate=True):
    d = df.copy()
    x = d["mLAM"].astype(float)
    y = d["mTT"].astype(float)
    lbl = d["rank"].astype(int).values

    colors = None
    if "group" in d.columns:
        def color_for_group(g):
            return GROUP_COLORS.get(str(g), plt.cm.tab20(3))
        colors = d["group"].astype(str).map(color_for_group).values

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(x, y, s=28, alpha=0.75, edgecolors="none", c=colors)

    if annotate:
        for xi, yi, la in zip(x, y, lbl):
            ax.text(float(xi), float(yi), str(la), fontsize=8,
                    ha="center", va="bottom", alpha=0.75, clip_on=True)

    ax.set_xlabel("mLAM")
    ax.set_ylabel("mTT")
    ax.grid(True, ls=":", alpha=0.4)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    if "group" in d.columns:
        uniq = list(pd.unique(d["group"].astype(str)))
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       color=GROUP_COLORS.get(g, plt.cm.tab20(3)), label=g)
            for g in uniq
        ]
        ax.legend(handles=handles, title="Group", loc="best")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_lam_tt_only(df, out_dir, prefix, exclude_ranks=None, annotate=True,
                     zoom_x=(0.6, 1.0), zoom_y=(0, 50)):
    d = df.copy()

    if exclude_ranks:
        exclude_ranks = set(exclude_ranks)
        d = d[~d["rank"].isin(exclude_ranks)].copy()

    dd = d.dropna(subset=["mLAM", "mTT"]).copy()

    pearson = dd["mLAM"].corr(dd["mTT"], method="pearson")
    spearman = dd["mLAM"].corr(dd["mTT"], method="spearman")
    print(f"[{prefix}] n={len(dd)}  Pearson={pearson:.4f}  Spearman={spearman:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_full = os.path.join(out_dir, f"{prefix}.png")
    out_zoom = os.path.join(out_dir, f"{prefix}_zoom.png")

    scatter_lam_tt(dd, out_full, title="LAM vs TT", annotate=annotate)
    scatter_lam_tt(dd, out_zoom, title="LAM vs TT (ZOOM)",
                   xlim=zoom_x, ylim=zoom_y, annotate=annotate)

    print(f"[Saved] {out_full}")
    print(f"[Saved] {out_zoom}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", default=MAIN_OUT_DIR)
    parser.add_argument("--vmin", type=int, default=2, help="垂直線/対角線の最小長（LAM/TT/DET用）")

    parser.add_argument("--exclude_ranks", type=str,
                        default=",".join(map(str, DEFAULT_EXCLUDE_RANKS)),
                        help="除外するrank番号（カンマ区切り）。例: 139,11,24")
    parser.add_argument("--no_annotate", action="store_true", help="散布図の注釈(rank)を消す")

    parser.add_argument("--group_method", choices=["tertile", "quartile", "threshold", "kmeans"],
                        default="kmeans", help="グルーピング手法（DEratio基準）")
    parser.add_argument("--threshold", type=str, default="",
                        help="group_method=threshold 用の境界（高->低）。例: 12,8")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise SystemExit(f"[ERROR] 入力フォルダ {args.dir} が存在しません")

    files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        raise SystemExit(f"[ERROR] {args.dir} に csv が見つかりません。")

    results = []
    for f in files:
        try:
            results.append(process_csv_for_metrics(f, vmin=args.vmin))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    df = pd.DataFrame(results)

    # ★DEratioがNaNの行はソート/グループできないので除外
    df = df.dropna(subset=["mDEratio"]).reset_index(drop=True)

    # ★DEratioでソート→rank
    df_sorted = df.sort_values(by="mDEratio", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)

    # ★DEratioでグルーピング
    thresholds = None
    if args.group_method == "threshold" and args.threshold.strip():
        thresholds = [float(s.strip()) for s in args.threshold.split(",") if s.strip()]

    df_grouped = assign_group(df_sorted, key="mDEratio", method=args.group_method, thresholds=thresholds)

    os.makedirs(args.out, exist_ok=True)
    df_grouped.to_csv(os.path.join(args.out, "rqa_summary_lam_tt.csv"), index=False)
    print(f"[Saved] {args.out}/rqa_summary_lam_tt.csv")

    # ★棒グラフもDEratio
    plot_sorted_bar(df_grouped, key="mDEratio",
                    path=os.path.join(args.out, "sorted_bar_mDEratio.png"))
    print(f"[Saved] {args.out}/sorted_bar_mDEratio.png")

    annotate = not args.no_annotate

    # 全点散布図（色はDEratio由来のgroup）
    plot_lam_tt_only(df_grouped, out_dir=args.out, prefix="lam_vs_tt",
                     exclude_ranks=None, annotate=annotate)

    # 外れ値除外
    ex = [int(s.strip()) for s in args.exclude_ranks.split(",") if s.strip()]
    plot_lam_tt_only(df_grouped, out_dir=args.out, prefix="lam_vs_tt_wo_outliers",
                     exclude_ranks=ex, annotate=annotate)

if __name__ == "__main__":
    main()
