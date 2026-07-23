import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 20,   # 軸ラベルサイズ
    "axes.titlesize": 22,   # タイトル
})


# ====== 設定 ======
INDICATOR_CSV  = "results_missing_compare.csv"
OUT_DIR        = "indicator_trends_by_missing"

MISSING_RATES = [0.0, 0.2, 0.5, 0.7, 0.9]
BASE_FEATURES = ["mDEratio", "mL", "mEN", "mTT", "mLAM"]

# 色（黄色は避ける）
INDICATOR_COLORS_BASE = {
    "Lorenz attractor (x)": "red",
    "Lorenz attractor (x) + noise": "orange",
    "Sine wave": "green",
    "Sine wave + noise": "goldenrod",   # ← yellow をやめる
    "Logistic map": "purple",
    "Logistic map + noise": "lightblue",
    "White noise": "black",
    "Brownian motion": "blue",
}

def norm(s: str) -> str:
    return " ".join(str(s).strip().split())

def get_indicator_color(name, all_indicators=None):
    key = norm(name)
    if key in INDICATOR_COLORS_BASE:
        return INDICATOR_COLORS_BASE[key]
    if all_indicators is None:
        return "k"
    idx = sorted(list(all_indicators)).index(name)
    return plt.cm.tab20(idx % 20)

def load_indicators(csv_path):
    h = pd.read_csv(csv_path)
    rename_map = {"DEratio":"mDEratio","L":"mL","L_entr":"mEN","TT":"mTT","LAM":"mLAM"}
    h = h.rename(columns={k: v for k, v in rename_map.items() if k in h.columns})

    if "indicator" not in h.columns:
        h["indicator"] = h["base_name"] if "base_name" in h.columns else h["name"]

    if "missing_rate" not in h.columns:
        raise ValueError("indicator csv に missing_rate 列がありません。")

    def _to_float_rate(v):
        if pd.isna(v): return np.nan
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"): return float(s[:-1]) / 100.0
            return float(s)
        return float(v)

    h["missing_rate"] = h["missing_rate"].map(_to_float_rate)
    h["indicator"] = h["indicator"].map(norm)  # 表記ゆれ対策
    return h

def build_indicator_means(h, feature_cols, missing_rates):
    hs = h.dropna(subset=["indicator", "missing_rate"]).copy()
    hs = hs[hs["missing_rate"].isin([float(x) for x in missing_rates])].copy()
    hs = hs.dropna(subset=feature_cols)

    means = (hs.groupby(["indicator", "missing_rate"], as_index=False)[feature_cols]
               .mean()
               .sort_values(["indicator", "missing_rate"]))
    return means

def plot_trends(means_df, feature_cols, missing_rates, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # xはカテゴリ表示（0%,20%,...）
    x_labels = [f"{int(r*100)}%" for r in missing_rates]
    x_pos = np.arange(len(missing_rates))
    x_map = {float(r): i for i, r in enumerate(missing_rates)}

    indicators = sorted(means_df["indicator"].unique())
    all_inds = set(indicators)

    # 点の重なり回避：指標ごとに少し横にずらす
    offsets = np.linspace(-0.22, 0.22, len(indicators))

    for feat in feature_cols:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)

        for j, ind in enumerate(indicators):
            sub = means_df[means_df["indicator"] == ind].copy()
            sub = sub.sort_values("missing_rate")

            xs0 = np.array([x_map[float(r)] for r in sub["missing_rate"] if float(r) in x_map], dtype=float)
            ys  = sub[feat].to_numpy(dtype=float)

            # 念のため長さを合わせる
            if len(xs0) != len(ys):
                ys = ys[:len(xs0)]

            xs = xs0 + offsets[j]
            c = get_indicator_color(ind, all_inds)

            # 線 + 点（点は白縁取り）
            ax.plot(xs, ys, linewidth=1.4, alpha=0.75, color=c, zorder=2)
            ax.scatter(xs, ys, s=58, alpha=0.95, color=c,
                       edgecolors="white", linewidths=0.8, zorder=3,
                       label=ind)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Missing rate (label)")
        ax.set_ylabel(feat)
        ax.grid(True, ls=":", alpha=0.4)

        # 凡例：2列で右外
        ax.legend(title="Indicator", ncol=2,
                  loc="lower right", 
                  borderaxespad=0.0, fontsize=9, title_fontsize=10)

        out_png = os.path.join(out_dir, f"trend_{feat}.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_png}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    h = load_indicators(INDICATOR_CSV)
    feature_cols = [c for c in BASE_FEATURES if c in h.columns]
    if len(feature_cols) == 0:
        raise SystemExit(f"[ERROR] 指標列が見つかりません: {BASE_FEATURES}")

    means = build_indicator_means(h, feature_cols, MISSING_RATES)
    means.to_csv(os.path.join(OUT_DIR, "indicator_means_by_missing_rate.csv"), index=False)

    plot_trends(means, feature_cols, MISSING_RATES, out_dir=os.path.join(OUT_DIR, "trends"))

if __name__ == "__main__":
    main()
