#実データと8指標を重ねて8つのグルーピングをするような処理

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====== 設定 ======
REALDATA_CSV   = "rqa_summary_sort.csv"
INDICATOR_CSV  = "results_missing_compare.csv"
OUT_DIR        = "nearest_indicator_result_by_missing"

# 欠損率（出したいもの）
MISSING_RATES = [0.0, 0.2, 0.5, 0.7, 0.9]

# 2Dで見たいペア（必要に応じて追加）
PLOT_PAIRS = [
    ("mDEratio", "mL"),
    ("mDEratio", "mEN"),
    ("mL", "mEN"),
    # ("mDEratio", "mTT"),
    # ("mL", "mTT"),
    # ("mEN", "mTT"),
    # ("mLAM", "mTT"),
]

# 指標色（ベース）
INDICATOR_COLORS_BASE = {
    "Lorenz attractor (x)": "red",
    "Sine wave":            "green",
    "Logistic map":         "purple",
    "White noise":          "black",
}
def get_indicator_color(name, all_indicators=None):
    if name in INDICATOR_COLORS_BASE:
        return INDICATOR_COLORS_BASE[name]
    if all_indicators is None:
        return "k"
    idx = sorted(list(all_indicators)).index(name)
    return plt.cm.tab20(idx % 20)


# ====== 指標CSVの読み込み＆整形 ======
def load_indicators(csv_path):
    """
    指標CSVを読み込み、列名を m*** に揃え、missing_rateをfloat化。
    """
    h = pd.read_csv(csv_path)

    rename_map = {
        "DEratio": "mDEratio",
        "L":       "mL",
        "L_entr":  "mEN",
        "TT":      "mTT",
        "LAM":     "mLAM",
    }
    h = h.rename(columns={k: v for k, v in rename_map.items() if k in h.columns})

    if "indicator" not in h.columns:
        if "base_name" in h.columns:
            h["indicator"] = h["base_name"]
        else:
            h["indicator"] = h["name"]

    # missing_rate 正規化
    if "missing_rate" in h.columns:
        def _to_float_rate(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, str):
                s = v.strip()
                if s.endswith("%"):
                    return float(s[:-1]) / 100.0
                return float(s)
            return float(v)
        h["missing_rate"] = h["missing_rate"].map(_to_float_rate)

    return h


# ====== 指標代表点（centroid）作成：欠損率指定 ======
def build_centroids_for_rate(h, feature_cols, missing_rate):
    """
    missing_rate を指定して、その欠損率の指標点から指標ごとの代表点（平均）を作る
    """
    if "missing_rate" not in h.columns:
        raise ValueError("indicator csv に missing_rate 列がありません。")

    hs = h[h["missing_rate"] == float(missing_rate)].copy()
    hs = hs.dropna(subset=["indicator"] + feature_cols)

    if len(hs) == 0:
        return None

    centroids = hs.groupby("indicator")[feature_cols].mean().reset_index()
    return centroids


# ====== 最近傍割り当て（標準化→距離） ======
def assign_nearest(df_real, centroids, feature_cols, standardize=True):
    d = df_real.copy()

    Xdf = d[feature_cols].astype(float)
    Cdf = centroids[feature_cols].astype(float)

    if standardize:
        # 実データで標準化パラメータを作る（欠損率ごとの比較がブレにくい）
        mu = Xdf.mean(axis=0)
        sd = Xdf.std(axis=0).replace(0, 1.0)
        X = ((Xdf - mu) / sd).to_numpy()
        C = ((Cdf - mu) / sd).to_numpy()
    else:
        X = Xdf.to_numpy()
        C = Cdf.to_numpy()

    dist = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = dist.argmin(axis=1)

    d["nearest_indicator"] = centroids["indicator"].iloc[nearest_idx].values
    d["nearest_dist"] = dist.min(axis=1)

    dist_df = pd.DataFrame(dist, columns=centroids["indicator"].tolist())
    return d, dist_df


# ====== 可視化：最近傍指標で色分けした散布図（指標★も描く） ======
def plot_grouped_pairs(df_assigned, centroids, pairs, out_path, title="",
                       save_each=False, each_dir=None):
    all_inds = set(centroids["indicator"].unique())

    def _plot_one(ax, xcol, ycol):
        # 実データ：nearest_indicatorごと
        for ind in sorted(df_assigned["nearest_indicator"].unique()):
            sub = df_assigned[df_assigned["nearest_indicator"] == ind]
            c = get_indicator_color(ind, all_inds)
            ax.scatter(sub[xcol], sub[ycol], s=24, alpha=0.8, c=[c],
                       edgecolors="none", label=ind)

        """
        # 指標代表点（★）
        for _, row in centroids.iterrows():
            ind = row["indicator"]
            c = get_indicator_color(ind, all_inds)
            ax.scatter(row[xcol], row[ycol], s=220, c=[c],
                       edgecolor="black", linewidths=0.9, zorder=3)
            #ax.text(row[xcol], row[ycol], ind, fontsize=8, fontweight="bold",ha="left", va="bottom", clip_on=True)
"""
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    # ===== まとめ図（1×n） =====
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (xcol, ycol) in zip(axes, pairs):
        _plot_one(ax, xcol, ycol)

  

    legend_handles = []
    legend_labels = []

    for ind in centroids["indicator"]:
        c = get_indicator_color(ind, all_inds)
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor=c, markersize=8)
        )
        legend_labels.append(ind)

    axes[-1].legend(
        legend_handles, legend_labels,
        title="Nearest indicator",
        loc="lower left",
        bbox_to_anchor=(0.40, 0.02),
        frameon=True
    )


    # 凡例（重複除去）
    
    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")

    # ===== 各ペアを1枚ずつ保存 =====
    if save_each:
        if each_dir is None:
            each_dir = os.path.join(os.path.dirname(out_path), "each_pairs")
        os.makedirs(each_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(out_path))[0]
        for (xcol, ycol) in pairs:
            one_out = os.path.join(each_dir, f"{base}_{xcol}_vs_{ycol}.png")

            fig1, ax1 = plt.subplots(1, 1, figsize=(5.2, 4.5), constrained_layout=True)
            _plot_one(ax1, xcol, ycol)

        
            # ===== 1枚図：全指標を必ず凡例に出す（centroidsベース）=====
            legend_handles_1 = []
            legend_labels_1 = []
            for ind in centroids["indicator"]:
                c = get_indicator_color(ind, all_inds)
                legend_handles_1.append(
                    Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=c, markersize=8)
                )
                legend_labels_1.append(ind)

            ax1.legend(legend_handles_1, legend_labels_1,
                    title="Nearest indicator",
                    loc="upper left")

            fig1.suptitle(f"{title} ({xcol} vs {ycol})")
            fig1.savefig(one_out, dpi=200, bbox_inches="tight")
        plt.close(fig1)

        print(f"[Saved] {one_out}")



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 実データ
    df = pd.read_csv(REALDATA_CSV)

    # 指標
    h = load_indicators(INDICATOR_CSV)

    # 特徴量：実データと指標の両方にある列だけ採用
    base_features = ["mDEratio", "mL", "mEN", "mTT", "mLAM"]
    feature_cols = [c for c in base_features if (c in df.columns and c in h.columns)]
    if len(feature_cols) < 2:
        raise SystemExit(f"[ERROR] 特徴量が不足: {feature_cols}")

    # 描画できるペアだけに絞る
    plot_pairs = [(x, y) for (x, y) in PLOT_PAIRS
                  if x in df.columns and y in df.columns and x in h.columns and y in h.columns]

    # 欠損率ごとに出力
    summary_rows = []

    for mr in MISSING_RATES:
        rate_dir = os.path.join(OUT_DIR, f"missing_{int(mr*100):02d}pct")
        os.makedirs(rate_dir, exist_ok=True)

        centroids = build_centroids_for_rate(h, feature_cols, mr)
        if centroids is None or len(centroids) == 0:
            print(f"[WARN] missing_rate={mr} の指標データが無いためスキップ")
            continue

        print(f"\n=== missing_rate={mr}  centroids={len(centroids)} ===")
        print("indicators:", centroids["indicator"].tolist())

        # 最近傍分類
        df_assigned, dist_df = assign_nearest(df, centroids, feature_cols, standardize=True)

        # 保存
        centroids.to_csv(os.path.join(rate_dir, "centroids.csv"), index=False)
        df_assigned.to_csv(os.path.join(rate_dir, "nearest_indicator_assignment.csv"), index=False)
        dist_df.to_csv(os.path.join(rate_dir, "distance_to_each_indicator.csv"), index=False)

        # 件数
        counts = df_assigned["nearest_indicator"].value_counts().reset_index()
        counts.columns = ["indicator", "count"]
        counts.to_csv(os.path.join(rate_dir, "counts_by_indicator.csv"), index=False)

        # 図
        if len(plot_pairs) > 0:
            out_png = os.path.join(rate_dir, "grouped_pairs.png")
            plot_grouped_pairs(
                df_assigned, centroids, plot_pairs, out_png,
                title=f"Real data grouped by nearest indicator (missing={int(mr*100)}%)",
                save_each=True,
                each_dir=os.path.join(rate_dir, "grouped_pairs_each")
            )


        # 全体サマリー用
        for _, r in counts.iterrows():
            summary_rows.append({
                "missing_rate": mr,
                "indicator": r["indicator"],
                "count": int(r["count"])
            })

    # 欠損率×指標の件数クロス表（比較しやすい）
    if len(summary_rows) > 0:
        summary = pd.DataFrame(summary_rows)
        pivot = summary.pivot_table(index="indicator", columns="missing_rate", values="count", fill_value=0)
        pivot.to_csv(os.path.join(OUT_DIR, "counts_pivot_by_missing_rate.csv"))
        summary.to_csv(os.path.join(OUT_DIR, "counts_long_by_missing_rate.csv"), index=False)
        print("[Saved] counts_pivot_by_missing_rate.csv")
        print("[Saved] counts_long_by_missing_rate.csv")

if __name__ == "__main__":
    main()
