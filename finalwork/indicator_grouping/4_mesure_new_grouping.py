#解析結果に4指標を加えたものとなっている
#処理時間短縮のため出力して得られた数値の記載してあるcsvファイルを読み込んで解析
#グラフを出力する
#解析結果におけるグルーピングの根拠付けのために存在



import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAIN_OUT_DIR = "rqa_result"   # 出力ルート

# ===== グループ色 =====
GROUP_COLORS = {
    "Low":  plt.cm.tab20(0),
    "Mid":  plt.cm.tab20(1),
    "High": plt.cm.tab20(2),
}
DEFAULT_COLOR = plt.cm.tab20(3)

# ===== 4指標の色・短縮ラベル =====
INDICATOR_COLORS = {
    "Lorenz attractor (x)": "red",
    "Sine wave":            "green",
    "Logistic map":         "purple",
    "White noise":          "black",
}
INDICATOR_SHORT = {
    "Lorenz attractor (x)": "Lorenz",
    "Sine wave":            "Sin",
    "Logistic map":         "Logi",
    "White noise":          "Noise",
}

# ---------------------------------------------------------
# df内の点を描く + 4指標を★で重ねる 2D ペア描画
# ---------------------------------------------------------
def scatter_pairs(d, pairs, out_path,
                  title="",
                  group_col="group",
                  rank_col="rank",
                  zoom_ranges=None,
                  figsize_scale=5,
                  annotate=True,
                  annotate_step=1,
                  highlight_df=None):
    n = len(pairs)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_scale*n, 6),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    if group_col not in d.columns:
        d[group_col] = "All"

    groups = d[group_col].astype("category")
    group_values = groups.values

    if rank_col in d.columns:
        labels_all = d[rank_col].values
    else:
        labels_all = np.arange(1, len(d)+1)

    def color_for_group(g):
        return GROUP_COLORS.get(str(g), DEFAULT_COLOR)

    colors_all = np.array([color_for_group(g) for g in group_values])

    used_indicator_labels = set()

    for ax_idx, (ax, (xcol, ycol)) in enumerate(zip(axes, pairs)):
        x = d[xcol].astype(float).values
        y = d[ycol].astype(float).values

        mask = np.ones_like(x, dtype=bool)

        if zoom_ranges and (xcol, ycol) in zoom_ranges:
            xr = zoom_ranges[(xcol, ycol)]["xlim"]
            yr = zoom_ranges[(xcol, ycol)]["ylim"]
            mask = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            xmin, xmax = xr
            ymin, ymax = yr
            is_zoom = True
        else:
            xmin = xmax = ymin = ymax = None
            is_zoom = False

        x_vis = x[mask]
        y_vis = y[mask]
        labels_vis = labels_all[mask]
        groups_vis = group_values[mask]
        colors_vis = colors_all[mask]

        for g in np.unique(groups_vis):
            gm = (groups_vis == g)
            ax.scatter(x_vis[gm], y_vis[gm],
                       c=colors_vis[gm],
                       s=24, alpha=0.8,
                       edgecolors="none",
                       label=str(g))

        if annotate:
            step = max(1, int(annotate_step))
            for i in range(0, len(x_vis), step):
                xi = float(x_vis[i])
                yi = float(y_vis[i])
                la = labels_vis[i]
                ax.text(xi, yi,
                        str(la),
                        fontsize=7,
                        ha="center", va="bottom",
                        alpha=0.7,
                        clip_on=True)

        # 4指標 ★
        if highlight_df is not None:
            h = highlight_df.copy()
            if xcol in h.columns and ycol in h.columns:
                if is_zoom:
                    h = h[(h[xcol] >= xmin) & (h[xcol] <= xmax) &
                          (h[ycol] >= ymin) & (h[ycol] <= ymax)]

                for _, row in h.iterrows():
                    name = row["indicator"]
                    xh = float(row[xcol])
                    yh = float(row[ycol])
                    cind = INDICATOR_COLORS.get(name, "k")
                    short = row["short_label"]

                    label = None
                    if ax_idx == 0 and name not in used_indicator_labels:
                        label = name
                        used_indicator_labels.add(name)

                    ax.scatter(xh, yh,
                               c=[cind], s=180, marker="*",
                               edgecolor="black", linewidths=0.8,
                               label=label)
                    ax.text(xh, yh,
                            short,
                            fontsize=10, fontweight="bold",
                            ha="left", va="bottom",
                            clip_on=True)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    axes[-1].legend(uniq.values(), uniq.keys(),
                    title="Group / Indicator", loc="best")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------
# 4指標だけの 2D グラフ（破損なし / 破損あり＋なし）
# ---------------------------------------------------------
def plot_4indicators_only_2d(indicator_df, out_root):
    if indicator_df is None or indicator_df.empty:
        return

    # 基本 3 ペア
    base_pairs = [("mDEratio", "mL"),
                  ("mDEratio", "mEN"),
                  ("mL", "mEN")]

    # TT を縦軸にした 3 ペア
    tt_pairs = [("mDEratio", "mTT"),
                ("mL", "mTT"),
                ("mEN", "mTT")]

    # LAM–TT（1 ペアだけ）
    lam_pair = ("mLAM", "mTT")

    out_dir = os.path.join(out_root, "4indicators_only")
    os.makedirs(out_dir, exist_ok=True)

    # ===============================
    # 1) 破損なし（missing_rate == 0%）
    # ===============================
    if "missing_rate" in indicator_df.columns:
        h0 = indicator_df[indicator_df["missing_rate"] == 0.0].copy()
    else:
        h0 = indicator_df.copy()

    if not h0.empty:
        # ---------- 基本 3 ペア ----------
        fig, axes = plt.subplots(
            1, len(base_pairs),
            figsize=(5 * len(base_pairs), 4.5)
        )
        if len(base_pairs) == 1:
            axes = [axes]

        for ax, (xcol, ycol) in zip(axes, base_pairs):
            for ind_name, sub in h0.groupby("indicator"):
                color = INDICATOR_COLORS.get(ind_name, "k")
                ax.scatter(sub[xcol], sub[ycol],
                           c=color, s=80, label=ind_name)
                for _, row in sub.iterrows():
                    ax.text(row[xcol], row[ycol],
                            row["short_label"],
                            fontsize=9, fontweight="bold",
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)

        axes[-1].legend(title="Indicator")
        fig.suptitle("4 indicators (missing 0%)", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(out_dir, "4indicators_nomissing_pairs.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {out_path}")

        # ---------- TT ペア ----------
        if {"mTT"} <= set(h0.columns):
            fig, axes = plt.subplots(
                1, len(tt_pairs),
                figsize=(5 * len(tt_pairs), 4.5)
            )
            if len(tt_pairs) == 1:
                axes = [axes]

            for ax, (xcol, ycol) in zip(axes, tt_pairs):
                for ind_name, sub in h0.groupby("indicator"):
                    color = INDICATOR_COLORS.get(ind_name, "k")
                    ax.scatter(sub[xcol], sub[ycol],
                               c=color, s=80, label=ind_name)
                    for _, row in sub.iterrows():
                        ax.text(row[xcol], row[ycol],
                                row["short_label"],
                                fontsize=9, fontweight="bold",
                                ha="left", va="bottom")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, ls=":", alpha=0.4)

            axes[-1].legend(title="Indicator")
            fig.suptitle("4 indicators (TT pairs, missing 0%)", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = os.path.join(out_dir, "4indicators_nomissing_TT_pairs.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"[Saved] {out_path}")

        # ---------- LAM–TT ----------
        if {"mLAM", "mTT"} <= set(h0.columns):
            fig, ax = plt.subplots(figsize=(5, 4.5))
            xcol, ycol = lam_pair
            for ind_name, sub in h0.groupby("indicator"):
                color = INDICATOR_COLORS.get(ind_name, "k")
                ax.scatter(sub[xcol], sub[ycol],
                           c=color, s=80, label=ind_name)
                for _, row in sub.iterrows():
                    ax.text(row[xcol], row[ycol],
                            row["short_label"],
                            fontsize=9, fontweight="bold",
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)
            ax.legend(title="Indicator")
            fig.suptitle("4 indicators (LAM vs TT, missing 0%)", y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = os.path.join(out_dir, "4indicators_nomissing_LAM_TT.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"[Saved] {out_path}")

    # ===============================
    # 2) 破損ありも含めて
    # ===============================
    h_all = indicator_df.copy()
    if "missing_rate" in h_all.columns:
        h_all["cond"] = h_all["missing_rate"].map(lambda r: f"{int(r*100)}%")
    else:
        h_all["cond"] = "0%"

    markers = {"0%": "o", "20%": "s", "50%": "^", "70%": "D"}

    # ---------- 基本 3 ペア ----------
    fig, axes = plt.subplots(
        1, len(base_pairs),
        figsize=(5 * len(base_pairs), 6)
    )
    if len(base_pairs) == 1:
        axes = [axes]

    for ax, (xcol, ycol) in zip(axes, base_pairs):
        for (ind_name, cond), sub in h_all.groupby(["indicator", "cond"]):
            color = INDICATOR_COLORS.get(ind_name, "k")
            marker = markers.get(cond, "o")
            label = f"{INDICATOR_SHORT.get(ind_name, ind_name)} ({cond})"
            ax.scatter(sub[xcol], sub[ycol],
                       c=color, marker=marker, s=70,
                       edgecolors="black", linewidths=0.5,
                       label=label)
            for _, row in sub.iterrows():
                ax.text(row[xcol], row[ycol],
                        row["short_label"],
                        fontsize=8,
                        ha="left", va="bottom")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h_, l_ in zip(handles, labels):
        uniq[l_] = h_
    axes[-1].legend(uniq.values(), uniq.keys(),
                    title="Indicator (missing rate)",
                    loc="best")

    fig.suptitle("4 indicators (with and without missing)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, "4indicators_with_missing_pairs.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")

    # ---------- TT ペア ----------
    if {"mTT"} <= set(h_all.columns):
        fig, axes = plt.subplots(
            1, len(tt_pairs),
            figsize=(5 * len(tt_pairs), 6)
        )
        if len(tt_pairs) == 1:
            axes = [axes]

        for ax, (xcol, ycol) in zip(axes, tt_pairs):
            for (ind_name, cond), sub in h_all.groupby(["indicator", "cond"]):
                color = INDICATOR_COLORS.get(ind_name, "k")
                marker = markers.get(cond, "o")
                label = f"{INDICATOR_SHORT.get(ind_name, ind_name)} ({cond})"
                ax.scatter(sub[xcol], sub[ycol],
                           c=color, marker=marker, s=70,
                           edgecolors="black", linewidths=0.5,
                           label=label)
                for _, row in sub.iterrows():
                    ax.text(row[xcol], row[ycol],
                            row["short_label"],
                            fontsize=8,
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        uniq = {}
        for h_, l_ in zip(handles, labels):
            uniq[l_] = h_
        axes[-1].legend(uniq.values(), uniq.keys(),
                        title="Indicator (missing rate)",
                        loc="best")

        fig.suptitle("4 indicators (TT pairs, with/without missing)", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(out_dir, "4indicators_with_missing_TT_pairs.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {out_path}")

    # ---------- LAM–TT ----------
    if {"mLAM", "mTT"} <= set(h_all.columns):
        fig, ax = plt.subplots(figsize=(5, 6))
        xcol, ycol = lam_pair
        for (ind_name, cond), sub in h_all.groupby(["indicator", "cond"]):
            color = INDICATOR_COLORS.get(ind_name, "k")
            marker = markers.get(cond, "o")
            label = f"{INDICATOR_SHORT.get(ind_name, ind_name)} ({cond})"
            ax.scatter(sub[xcol], sub[ycol],
                       c=color, marker=marker, s=70,
                       edgecolors="black", linewidths=0.5,
                       label=label)
            for _, row in sub.iterrows():
                ax.text(row[xcol], row[ycol],
                        row["short_label"],
                        fontsize=8,
                        ha="left", va="bottom")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h_, l_ in zip(handles, labels):
            uniq[l_] = h_
        ax.legend(uniq.values(), uniq.keys(),
                  title="Indicator (missing rate)",
                  loc="best")

        fig.suptitle("4 indicators (LAM vs TT, with/without missing)", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(out_dir, "4indicators_with_missing_LAM_TT.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {out_path}")
 
def plot_4indicators_only_TT_LAM_by_missing(indicator_df, out_root):
    """
    4 指標のみを対象に、
      - DEratio ペア (mDEratio–mL, mDEratio–mEN, mL–mEN)
      - TT ペア     (mDEratio–mTT, mL–mTT, mEN–mTT)
      - LAM–TT      (mLAM–mTT)
    を missing_rate ごとに出力する。

    実データ（df_all）は使わない。
    出力先: out_root / "4indicators_only_by_missing" / "missing_XXpct"
    """
    if indicator_df is None or indicator_df.empty:
        return
    if "missing_rate" not in indicator_df.columns:
        return

    # DEratio ベースの 3 ペア
    de_pairs = [("mDEratio", "mL"),
                ("mDEratio", "mEN"),
                ("mL", "mEN")]

    # TT ペア
    tt_pairs = [("mDEratio", "mTT"),
                ("mL", "mTT"),
                ("mEN", "mTT")]

    # LAM–TT
    lam_pair = ("mLAM", "mTT")

    base_dir = os.path.join(out_root, "4indicators_only_by_missing")
    os.makedirs(base_dir, exist_ok=True)

    for r in sorted(indicator_df["missing_rate"].unique()):
        sub = indicator_df[indicator_df["missing_rate"] == r].copy()
        if sub.empty:
            continue

        rate_label = f"{int(r*100)}%"
        print(f"\n=== 4 indicators only (DE / TT / LAM–TT), missing {rate_label} ===")

        # 念のため indicator / short_label が無ければ作る
        if "indicator" not in sub.columns:
            if "base_name" in sub.columns:
                sub["indicator"] = sub["base_name"]
            else:
                sub["indicator"] = sub["name"]
        if "short_label" not in sub.columns:
            sub["short_label"] = sub["indicator"].map(INDICATOR_SHORT)

        out_dir = os.path.join(base_dir, f"missing_{int(r*100)}pct")
        os.makedirs(out_dir, exist_ok=True)

        # ---------- DEratio ペア ----------
        if {"mDEratio", "mL", "mEN"} <= set(sub.columns):
            fig, axes = plt.subplots(
                1, len(de_pairs),
                figsize=(5 * len(de_pairs), 4.5)
            )
            if len(de_pairs) == 1:
                axes = [axes]

            for ax, (xcol, ycol) in zip(axes, de_pairs):
                for ind_name, g in sub.groupby("indicator"):
                    color = INDICATOR_COLORS.get(ind_name, "k")
                    ax.scatter(g[xcol], g[ycol],
                               c=color, s=80, label=ind_name)
                    for _, row in g.iterrows():
                        ax.text(row[xcol], row[ycol],
                                row["short_label"],
                                fontsize=9, fontweight="bold",
                                ha="left", va="bottom")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, ls=":", alpha=0.4)

            # 凡例をユニークに
            handles, labels = axes[0].get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            axes[-1].legend(uniq.values(), uniq.keys(), title="Indicator")

            fig.suptitle(
                f"4 indicators only (DEratio pairs, missing {rate_label})",
                y=0.98
            )
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_de = os.path.join(
                out_dir,
                f"4indicators_only_DE_pairs_missing_{int(r*100)}pct.png"
            )
            fig.savefig(f_de, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_de}")

        # ---------- TT ペア ----------
        if {"mTT"} <= set(sub.columns):
            fig, axes = plt.subplots(
                1, len(tt_pairs),
                figsize=(5 * len(tt_pairs), 4.5)
            )
            if len(tt_pairs) == 1:
                axes = [axes]

            for ax, (xcol, ycol) in zip(axes, tt_pairs):
                for ind_name, g in sub.groupby("indicator"):
                    color = INDICATOR_COLORS.get(ind_name, "k")
                    ax.scatter(g[xcol], g[ycol],
                               c=color, s=80, label=ind_name)
                    for _, row in g.iterrows():
                        ax.text(row[xcol], row[ycol],
                                row["short_label"],
                                fontsize=9, fontweight="bold",
                                ha="left", va="bottom")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, ls=":", alpha=0.4)

            handles, labels = axes[0].get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            axes[-1].legend(uniq.values(), uniq.keys(), title="Indicator")

            fig.suptitle(
                f"4 indicators only (TT pairs, missing {rate_label})",
                y=0.98
            )
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_tt = os.path.join(
                out_dir,
                f"4indicators_only_TT_pairs_missing_{int(r*100)}pct.png"
            )
            fig.savefig(f_tt, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_tt}")

        # ---------- LAM–TT ----------
        if {"mLAM", "mTT"} <= set(sub.columns):
            xcol, ycol = lam_pair
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for ind_name, g in sub.groupby("indicator"):
                color = INDICATOR_COLORS.get(ind_name, "k")
                ax.scatter(g[xcol], g[ycol],
                           c=color, s=80, label=ind_name)
                for _, row in g.iterrows():
                    ax.text(row[xcol], row[ycol],
                            row["short_label"],
                            fontsize=9, fontweight="bold",
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)

            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h_, l_ in zip(handles, labels):
                uniq[l_] = h_
            ax.legend(uniq.values(), uniq.keys(), title="Indicator")

            fig.suptitle(
                f"4 indicators only (LAM vs TT, missing {rate_label})",
                y=0.98
            )
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            f_lam = os.path.join(
                out_dir,
                f"4indicators_only_LAM_TT_missing_{int(r*100)}pct.png"
            )
            fig.savefig(f_lam, dpi=200)
            plt.close(fig)
            print(f"[Saved] {f_lam}")


# ---------------------------------------------------------
# 4指標＋実データを重ねた 2D グラフ
# ---------------------------------------------------------
def plot_4indicators_with_background_2d(df_all, indicator_df, out_root):
    if indicator_df is None or indicator_df.empty:
        return

    base_pairs = [("mDEratio", "mL"),
                  ("mDEratio", "mEN"),
                  ("mL", "mEN")]

    out_dir = os.path.join(out_root, "4indicators_with_background")
    os.makedirs(out_dir, exist_ok=True)

    d = df_all.copy()
    if "group" not in d.columns:
        d["group"] = "All"

    groups = d["group"].astype("category").values

    def color_for_group(g):
        return GROUP_COLORS.get(str(g), DEFAULT_COLOR)

    colors = np.array([color_for_group(g) for g in groups])

    # 1) 破損なし
    if "missing_rate" in indicator_df.columns:
        h0 = indicator_df[indicator_df["missing_rate"] == 0.0].copy()
    else:
        h0 = indicator_df.copy()

    if not h0.empty:
        fig, axes = plt.subplots(
            1, len(base_pairs),
            figsize=(5 * len(base_pairs), 4)
        )
        if len(base_pairs) == 1:
            axes = [axes]

        for ax, (xcol, ycol) in zip(axes, base_pairs):
            ax.scatter(d[xcol].astype(float),
                       d[ycol].astype(float),
                       c=colors,
                       s=20, alpha=0.4,
                       edgecolors="none")
            for ind_name, sub in h0.groupby("indicator"):
                color = INDICATOR_COLORS.get(ind_name, "k")
                ax.scatter(sub[xcol], sub[ycol],
                           c=color, s=80,
                           edgecolors="black", linewidths=0.6,
                           label=ind_name)
                for _, row in sub.iterrows():
                    ax.text(row[xcol], row[ycol],
                            row["short_label"],
                            fontsize=9, fontweight="bold",
                            ha="left", va="bottom")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, ls=":", alpha=0.4)

        axes[-1].legend(title="Indicator (0%)")
        fig.suptitle("4 indicators + data (missing 0%)", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(
            out_dir, "4indicators_nomissing_pairs_withdata.png"
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {out_path}")

    # 2) 破損ありも含めて
    h_all = indicator_df.copy()
    if "missing_rate" in h_all.columns:
        h_all["cond"] = h_all["missing_rate"].map(lambda r: f"{int(r*100)}%")
    else:
        h_all["cond"] = "0%"

    markers = {"0%": "o", "20%": "s", "50%": "^", "70%": "D"}

    fig, axes = plt.subplots(
        1, len(base_pairs),
        figsize=(5 * len(base_pairs), 4)
    )
    if len(base_pairs) == 1:
        axes = [axes]

    for ax, (xcol, ycol) in zip(axes, base_pairs):
        ax.scatter(d[xcol].astype(float),
                   d[ycol].astype(float),
                   c=colors,
                   s=20, alpha=0.4,
                   edgecolors="none")

        for (ind_name, cond), sub in h_all.groupby(["indicator", "cond"]):
            color = INDICATOR_COLORS.get(ind_name, "k")
            marker = markers.get(cond, "o")
            label = f"{INDICATOR_SHORT.get(ind_name, ind_name)} ({cond})"
            ax.scatter(sub[xcol], sub[ycol],
                       c=color, marker=marker, s=70,
                       edgecolors="black", linewidths=0.6,
                       label=label)
            for _, row in sub.iterrows():
                ax.text(row[xcol], row[ycol],
                        row["short_label"],
                        fontsize=8,
                        ha="left", va="bottom")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h_, l_ in zip(handles, labels):
        uniq[l_] = h_
    axes[-1].legend(uniq.values(), uniq.keys(),
                    title="Indicator (missing rate)",
                    loc="best")

    fig.suptitle("4 indicators + data (with/without missing)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(
        out_dir, "4indicators_withmissing_pairs_withdata.png"
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")

def plot_4indicators_param_space_by_missing(indicator_df, df_all, out_root):
    """
    results_missing_compare.csv から読んだ 4 指標 (indicator_df) と
    実データ df_all を使って、

        missing_rate ごとに
        3D / 2D / ZOOM / TT / LAM–TT

    を出力する。
    """
    if indicator_df is None or indicator_df.empty:
        return
    if "missing_rate" not in indicator_df.columns:
        return

    base_dir = os.path.join(out_root, "4indicators_param_space_by_missing")
    os.makedirs(base_dir, exist_ok=True)

    # 実データ
    d = df_all.copy()
    if "group" not in d.columns:
        d["group"] = "All"

    for r in sorted(indicator_df["missing_rate"].unique()):
        sub = indicator_df[indicator_df["missing_rate"] == r].copy()
        if sub.empty:
            continue

        rate_label = f"{int(r*100)}%"
        print(f"\n=== plotting param space for missing {rate_label} ===")

        # 4 指標側に short_label / indicator が入っている前提
        # （まだならここで追加しておく）
        if "indicator" not in sub.columns:
            # base_name or name から indicator を決める
            if "base_name" in sub.columns:
                sub["indicator"] = sub["base_name"]
            else:
                sub["indicator"] = sub["name"]
        if "short_label" not in sub.columns:
            sub["short_label"] = sub["indicator"].map(INDICATOR_SHORT)

        # 出力フォルダ（missing 率ごとに分ける）
        out_r = os.path.join(base_dir, f"missing_{int(r*100)}pct")

        # 実データ d + 4 指標 sub を使って、さきほどの共通関数を呼ぶ
        _plot_param_space_core(
            d,
            highlight_df=sub,
            out_root=out_r,
            title_suffix=f" (missing {rate_label})"
        )


def _plot_param_space_core(d, highlight_df, out_root, title_suffix=""):
    """
    df 全体 d と、重ねたい 4 指標 highlight_df を使って
    3D / 2D / ZOOM / TT / LAM–TT を out_root に保存する共通関数。

    title_suffix には " (missing 20%)" のような文字列を渡す。
    """
    # 既存フォルダは消して作り直す
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    # 列チェック
    need = {"mDEratio", "mL", "mEN"}
    if not need <= set(d.columns):
        missing = need - set(d.columns)
        raise SystemExit(f"[ERROR] 必須列が不足: {missing}")

    # group 無ければダミー
    if "group" not in d.columns:
        d = d.copy()
        d["group"] = "All"

    # ========= 3D =========
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        groups = d["group"].astype("category")
        def color_for_group(g):
            return GROUP_COLORS.get(str(g), DEFAULT_COLOR)
        colors = [color_for_group(g) for g in groups]

        ax.scatter(d["mDEratio"].astype(float),
                   d["mL"].astype(float),
                   d["mEN"].astype(float),
                   c=colors, s=40, depthshade=True)

        # 4 指標（重ねたいもの）をプロット
        if (highlight_df is not None) and (not highlight_df.empty):
            used = set()
            for _, row in highlight_df.iterrows():
                name = row["indicator"]
                cind = INDICATOR_COLORS.get(name, "k")
                label = name if name not in used else None
                used.add(name)

                ax.scatter(float(row["mDEratio"]),
                           float(row["mL"]),
                           float(row["mEN"]),
                           c=[cind], s=180, marker="*",
                           edgecolor="black", linewidths=0.8,
                           label=label)
                ax.text(float(row["mDEratio"]),
                        float(row["mL"]),
                        float(row["mEN"]),
                        INDICATOR_SHORT.get(name, name),
                        fontsize=9, fontweight="bold")

        ax.set_xlabel("mDEratio")
        ax.set_ylabel("mL")
        ax.set_zlabel("mEN")
        ax.set_title("RQA Parameter Space (3D)" + title_suffix)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc="best")

        f3d = os.path.join(out_root, "rqa_param_space_3d.png")
        fig.savefig(f3d, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {f3d}")
    except Exception as e:
        print("[WARN] 3D plot skipped:", e)

    # ========= 2D base pairs =========
    base_pairs = [("mDEratio", "mL"),
                  ("mDEratio", "mEN"),
                  ("mL", "mEN")]

    zoom_ranges_base = {
        ("mDEratio", "mL"):  {"xlim": (5, 24), "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    }

    out_full = os.path.join(out_root, "rqa_param_space_pairs.png")
    scatter_pairs(d, base_pairs,
                  out_path=out_full,
                  title="RQA Parameter Space (2D pairs)" + title_suffix,
                  zoom_ranges=None,
                  annotate=True,
                  annotate_step=1,
                  highlight_df=highlight_df)

    out_zoom = os.path.join(out_root, "rqa_param_space_pairs_zoom.png")
    scatter_pairs(d, base_pairs,
                  out_path=out_zoom,
                  title="RQA Parameter Space (2D pairs, ZOOM)" + title_suffix,
                  zoom_ranges=zoom_ranges_base,
                  annotate=True,
                  annotate_step=1,
                  highlight_df=highlight_df)

    # ========= TT ペア =========
    if {"mTT"} <= set(d.columns):
        tt_pairs = [("mDEratio", "mTT"),
                    ("mL", "mTT"),
                    ("mEN", "mTT")]

        zoom_tt = {
            ("mDEratio", "mTT"): {"xlim": (10, 25), "ylim": (0, 45)},
            ("mL", "mTT"):       {"xlim": (0, 25),  "ylim": (0, 40)},
            ("mEN", "mTT"):      {"xlim": (0, 3),   "ylim": (0, 40)},
        }

        out_tt = os.path.join(out_root, "rqa_param_space_pairs_TT.png")
        scatter_pairs(d, tt_pairs,
                      out_path=out_tt,
                      title="RQA Parameter Space (TT pairs)" + title_suffix,
                      zoom_ranges=None,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

        out_tt_zoom = os.path.join(out_root, "rqa_param_space_pairs_TT_zoom.png")
        scatter_pairs(d, tt_pairs,
                      out_path=out_tt_zoom,
                      title="RQA Parameter Space (TT pairs, ZOOM)" + title_suffix,
                      zoom_ranges=zoom_tt,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

    # ========= LAM vs TT =========
    if {"mLAM", "mTT"} <= set(d.columns):
        lam_pairs = [("mLAM", "mTT")]
        zoom_lam = {
            ("mLAM", "mTT"): {"xlim": (0.6, 1.0), "ylim": (0, 50)},
        }

        out_lam = os.path.join(out_root, "rqa_param_space_pairs_LAM_TT.png")
        scatter_pairs(d, lam_pairs,
                      out_path=out_lam,
                      title="RQA Parameter Space (LAM vs TT)" + title_suffix,
                      zoom_ranges=None,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

        out_lam_zoom = os.path.join(out_root, "rqa_param_space_pairs_LAM_TT_zoom.png")
        scatter_pairs(d, lam_pairs,
                      out_path=out_lam_zoom,
                      title="RQA Parameter Space (LAM vs TT, ZOOM)" + title_suffix,
                      zoom_ranges=zoom_lam,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

    # ========= CSV =========
    cols = ["file", "group", "rank",
            "mDEratio", "mL", "mEN", "mTT", "mLAM"]
    cols = [c for c in cols if c in d.columns]
    out_csv = os.path.join(out_root, "rqa_param_space_points.csv")
    d[cols].to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

# ---------------------------------------------------------
# メイン：実データ + 4指標
# ---------------------------------------------------------
def plot_rqa_param_space(df, out_root=MAIN_OUT_DIR):
    # 実データ（rqa_summary_sort.csv）
    d = df.copy()

    need = {"mDEratio", "mL", "mEN"}
    if not need <= set(d.columns):
        missing = need - set(d.columns)
        raise SystemExit(f"[ERROR] 必須列が不足: {missing}")

    if "group" not in d.columns:
        d["group"] = "All"

    # ===== 4 指標 CSV 読み込み =====
    highlight_df = None      # 破損なしのみ（全体の Param Space に重ねる用）
    indicator_full_df = None # 破損あり・なし全部（4 指標の可視化用）

    # 新しいファイルがあればそれを優先
    if os.path.exists("results_missing_compare.csv"):
        indicator_csv = "results_missing_compare.csv"
    elif os.path.exists("results.csv"):
        indicator_csv = "results.csv"
    else:
        indicator_csv = None

    if indicator_csv is not None:
        try:
            h = pd.read_csv(indicator_csv)
            rename_map_h = {
                "DEratio": "mDEratio",
                "L":       "mL",
                "L_entr":  "mEN",
                "TT":      "mTT",
                "LAM":     "mLAM",
            }
            h = h.rename(columns={k: v for k, v in rename_map_h.items()
                                  if k in h.columns})

            # 4 指標だけ取り出す
            if "base_name" in h.columns:
                h["indicator"] = h["base_name"]
            else:
                h["indicator"] = h["name"]
            h = h[h["indicator"].isin(INDICATOR_COLORS.keys())].copy()

            # short_label を付与
            h["short_label"] = h["indicator"].map(INDICATOR_SHORT)

            indicator_full_df = h.copy()

            # highlight 用は「破損なしだけ」に絞る（missing_rate列があれば）
            if "missing_rate" in h.columns:
                highlight_df = h[h["missing_rate"] == 0.0].copy()
            else:
                highlight_df = h.copy()

            print(f"Loaded 4 indicators from {indicator_csv}")

        except Exception as e:
            print("[WARN] failed to load 4-indicator csv:", e)
    else:
        print("[INFO] no indicator csv found; skip 4-indicator overlay.")

    # ===== 実データ + 4 指標（破損なし）の Param Space 一式 =====
    _plot_param_space_core(d, highlight_df, out_root, title_suffix="")

    # ===== 4 指標だけの 2D 図 / 実データとの重ね図 / 破損率ごとの Param Space =====
    if indicator_full_df is not None and not indicator_full_df.empty:
        # 4 指標のみの 2D（0% / 0+20+50+70%）
        plot_4indicators_only_2d(indicator_full_df, out_root)

        # 実データを背景に、全 missing を重ねた 2D
        plot_4indicators_with_background_2d(d, indicator_full_df, out_root)

        # 破損率ごとに、3D / 2D / ZOOM / TT / LAM–TT を出力
        if "missing_rate" in indicator_full_df.columns:
            plot_4indicators_param_space_by_missing(indicator_full_df, d, out_root)
            plot_4indicators_only_TT_LAM_by_missing(indicator_full_df, out_root)


if __name__ == "__main__":
    df = pd.read_csv("rqa_summary_sort.csv")
    plot_rqa_param_space(df, out_root="4_grouping_result")
