# 2025-12-15
#ローレンツアトラクタ、ロジスティックス写像、sin波、ホワイトノイズに関して指標を決めRQAしたもの
#指標それぞれに関する値を出力する
#時系列の長さとしてすべて2000に固定して処理している
#指標それぞれに関して20%、50%、70%、90%の破損を加えて線形補間を施している
#破損後の時系列も出力している


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAIN_OUT_DIR = "8_grouping_result"   # 出力ルート

# ここはあなたのコードからそのまま
INDICATOR_COLORS_BASE = {
    "Lorenz attractor (x)": "red",
    "Sine wave":            "green",
    "Logistic map":         "purple",
    "White noise":          "black",
}
INDICATOR_SHORT_BASE = {
    "Lorenz attractor (x)": "Lorenz",
    "Sine wave":            "Sin",
    "Logistic map":         "Logi",
    "White noise":          "Noise",
}

def get_indicator_color(name, all_indicators=None):
    if name in INDICATOR_COLORS_BASE:
        return INDICATOR_COLORS_BASE[name]
    if all_indicators is None:
        return "k"
    idx = sorted(list(all_indicators)).index(name)
    return plt.cm.tab20(idx % 20)

def get_indicator_short(name):
    return INDICATOR_SHORT_BASE.get(name, name)

def plot_indicators_only_TT_LAM_by_indicator(indicator_df, out_root):
    """
    アップロードした指標 csv から、
      - DEratio ペア (mDEratio–mL, mDEratio–mEN, mL–mEN)
      - TT ペア     (mDEratio–mTT, mL–mTT, mEN–mTT)
      - LAM–TT      (mLAM–mTT)
    を「指標ごと」に 2D 図として出力する。
    1つの図の中では missing_rate ごとにマーカーを変えてプロットする。
    """
    if indicator_df is None or indicator_df.empty:
        return
    if "missing_rate" not in indicator_df.columns:
        return

    de_pairs = [("mDEratio", "mL"),
                ("mDEratio", "mEN"),
                ("mL", "mEN")]

    tt_pairs = [("mDEratio", "mTT"),
                ("mL", "mTT"),
                ("mEN", "mTT")]

    lam_pair = ("mLAM", "mTT")

    base_dir = os.path.join(out_root, "indicators_only_by_indicator")
    os.makedirs(base_dir, exist_ok=True)

    indicator_df = indicator_df.copy()
    indicator_df["cond"] = indicator_df["missing_rate"].map(
        lambda r: f"{int(r*100)}%"
    )

    markers = {"0%": "o", "20%": "s", "50%": "^", "70%": "D"}

    all_indicators = set(indicator_df["indicator"].unique())

    for ind_name, sub in indicator_df.groupby("indicator"):
        if sub.empty:
            continue

        sub = sub.copy()
        if "short_label" not in sub.columns:
            sub["short_label"] = get_indicator_short(ind_name)

        short = get_indicator_short(ind_name)

        out_dir = os.path.join(
            base_dir,
            f"indicator_{short}".replace(" ", "_")
        )
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Indicator {ind_name} (DE / TT / LAM–TT, all missing_rate) ===")

        # ----- DEratio ペア -----
        if {"mDEratio", "mL", "mEN"} <= set(sub.columns):
            fig, axes = plt.subplots(1, len(de_pairs),
                                     figsize=(5 * len(de_pairs), 4.5))
            if len(de_pairs) == 1:
                axes = [axes]

            for ax, (xcol, ycol) in zip(axes, de_pairs):
                for cond, g in sub.groupby("cond"):
                    marker = markers.get(cond, "o")
                    color = get_indicator_color(ind_name, all_indicators)
                    label = f"{short} ({cond})"
                    ax.scatter(g[xcol], g[ycol],
                               color=color, marker=marker, s=80,
                               edgecolors="black", linewidths=0.5,
                               label=label)
                    for _, row in g.iterrows():
                        rate_label = f"{int(row['missing_rate']*100)}%"
                        ax.text(row[xcol], row[ycol],
                                rate_label,
                                fontsize=9, fontweight="bold",
                                ha="left", va="bottom")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, ls=":", alpha=0.4)

            handles, labels = axes[0].get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            axes[-1].legend(uniq.values(), uniq.keys(),
                            title=f"{short} (missing rate)")

            fig.suptitle(f"{ind_name} (DEratio pairs, all missing_rate)", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_de = os.path.join(out_dir, f"{short}_DE_pairs_all_missing.png")
            fig.savefig(f_de, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_de}")

        # ----- TT ペア -----
        if {"mTT"} <= set(sub.columns):
            fig, axes = plt.subplots(1, len(tt_pairs),
                                     figsize=(5 * len(tt_pairs), 4.5))
            if len(tt_pairs) == 1:
                axes = [axes]

            for ax, (xcol, ycol) in zip(axes, tt_pairs):
                for cond, g in sub.groupby("cond"):
                    marker = markers.get(cond, "o")
                    color = get_indicator_color(ind_name, all_indicators)
                    label = f"{short} ({cond})"
                    ax.scatter(g[xcol], g[ycol],
                               color=color, marker=marker, s=80,
                               edgecolors="black", linewidths=0.5,
                               label=label)
                    for _, row in g.iterrows():
                        rate_label = f"{int(row['missing_rate']*100)}%"
                        ax.text(row[xcol], row[ycol],
                                rate_label,
                                fontsize=9, fontweight="bold",
                                ha="left", va="bottom")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, ls=":", alpha=0.4)

            handles, labels = axes[0].get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            axes[-1].legend(uniq.values(), uniq.keys(),
                            title=f"{short} (missing rate)")

            fig.suptitle(f"{ind_name} (TT pairs, all missing_rate)", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_tt = os.path.join(out_dir, f"{short}_TT_pairs_all_missing.png")
            fig.savefig(f_tt, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_tt}")

        # ----- LAM–TT -----
        if {"mLAM", "mTT"} <= set(sub.columns):
            xcol, ycol = lam_pair
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for cond, g in sub.groupby("cond"):
                marker = markers.get(cond, "o")
                color = get_indicator_color(ind_name, all_indicators)
                label = f"{short} ({cond})"
                ax.scatter(g[xcol], g[ycol],
                           color=color, marker=marker, s=80,
                           edgecolors="black", linewidths=0.5,
                           label=label)
                for _, row in g.iterrows():
                    rate_label = f"{int(row['missing_rate']*100)}%"
                    ax.text(row[xcol], row[ycol],
                            rate_label,
                            fontsize=9, fontweight="bold",
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)

            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            ax.legend(uniq.values(), uniq.keys(),
                      title=f"{short} (missing rate)")

            fig.suptitle(f"{ind_name} (LAM vs TT, all missing_rate)", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_lam = os.path.join(out_dir, f"{short}_LAM_TT_all_missing.png")
            fig.savefig(f_lam, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_lam}")


if __name__ == "__main__":
    # ★ここを「アップロードした csv のパス」に合わせて書き換える
    csv_path = "results_missing_compare.csv"  # 例

    h = pd.read_csv(csv_path)

    # 列名を m*** に揃える（元 csv の列名が DEratio, L, L_entr, TT, LAM の場合）
    rename_map_h = {
        "DEratio": "mDEratio",
        "L":       "mL",
        "L_entr":  "mEN",
        "TT":      "mTT",
        "LAM":     "mLAM",
    }
    h = h.rename(columns={k: v for k, v in rename_map_h.items() if k in h.columns})

    # indicator 列を作る（base_name or name から）
    if "indicator" not in h.columns:
        if "base_name" in h.columns:
            h["indicator"] = h["base_name"]
        else:
            h["indicator"] = h["name"]

    # short_label がなければ追加
    if "short_label" not in h.columns:
        h["short_label"] = h["indicator"].map(get_indicator_short)

    # 8指標すべてを使いたいので「4つだけに絞るフィルタ」はしない
    # h = h[h["indicator"].isin(INDICATOR_COLORS_BASE.keys())]  # ←これは削除してOK

    plot_indicators_only_TT_LAM_by_indicator(h, out_root=MAIN_OUT_DIR)
