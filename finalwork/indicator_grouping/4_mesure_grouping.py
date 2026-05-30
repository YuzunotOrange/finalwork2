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
# 先に zoom 範囲でマスク → その範囲だけ描く scatter_pairs
#   - zoom_ranges が None: 全点描画（全体図）
#   - zoom_ranges が dict: そのペアの範囲だけ描画（拡大図）
#   - clip_on=True で枠外に数字が飛び出さない
#   - highlight_df に 4 指標を渡すと ★ で重ね描き
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
    """
    d : DataFrame
    pairs : [("xcol","ycol"), ...]
    zoom_ranges : {("xcol","ycol"): {"xlim":(a,b), "ylim":(c,d)}, ...} or None
    annotate : True のとき点の番号を描く
    annotate_step : 何点ごとに番号を描くか（混雑対策）
    highlight_df : 4指標の DataFrame（列 indicator, short_label, mDEratio 等）
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n,
                             figsize=(figsize_scale*n, 4.5),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    # グループ列が無ければダミー
    if group_col not in d.columns:
        d[group_col] = "All"

    groups = d[group_col].astype("category")
    group_values = groups.values

    # rank が無ければ 1..N
    if rank_col in d.columns:
        labels_all = d[rank_col].values
    else:
        labels_all = np.arange(1, len(d)+1)

    # group → color
    def color_for_group(g):
        return GROUP_COLORS.get(str(g), DEFAULT_COLOR)

    colors_all = np.array([color_for_group(g) for g in group_values])

    # 4指標の凡例ラベルが重複しないように管理
    used_indicator_labels = set()

    for ax_idx, (ax, (xcol, ycol)) in enumerate(zip(axes, pairs)):
        x = d[xcol].astype(float).values
        y = d[ycol].astype(float).values

        # --- デフォルト: 全点可視 ---
        mask = np.ones_like(x, dtype=bool)

        # --- zoom_ranges があれば先にマスク ---
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

        # 可視範囲のみ抽出（ここが「先にマスク」部分）
        x_vis = x[mask]
        y_vis = y[mask]
        labels_vis = labels_all[mask]
        groups_vis = group_values[mask]
        colors_vis = colors_all[mask]

        # ---- グループごとに散布 ----
        for g in np.unique(groups_vis):
            gm = (groups_vis == g)
            ax.scatter(x_vis[gm], y_vis[gm],
                       c=colors_vis[gm],
                       s=24, alpha=0.8,
                       edgecolors="none",
                       label=str(g))

        # ---- 背景点の番号ラベル ----
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
                        clip_on=True)  # 枠外に出ない

        # ---- 4指標 ★ の描画 ----
        if highlight_df is not None:
            h = highlight_df.copy()

            # このペアが持っている列だけ使う（mTT, mLAM など）
            if xcol not in h.columns or ycol not in h.columns:
                pass
            else:
                # zoom 範囲があれば、その中に入る指標だけ描く
                if is_zoom:
                    h = h[(h[xcol] >= xmin) & (h[xcol] <= xmax) &
                          (h[ycol] >= ymin) & (h[ycol] <= ymax)]

                for _, row in h.iterrows():
                    name = row["indicator"]
                    xh = float(row[xcol])
                    yh = float(row[ycol])
                    cind = INDICATOR_COLORS.get(name, "k")
                    short = row["short_label"]

                    # 凡例は最初の軸で一度だけ
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

    # ---- 共同凡例（グループ＋4指標） ----
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
# df 全体から 3D / 2D / 拡大図 をまとめて出す関数
#   - 3つ並びの2D図（mDEratio–mL, mDEratio–mEN, mL–mEN）
#   - 3つ並びの TT ペア
#   - LAM vs TT（1枚）
#   - それぞれ全体図＋ZOOM図
#   - 4指標を results.csv から読み込み、重ね描き
# ---------------------------------------------------------
def plot_rqa_param_space(df, out_root=MAIN_OUT_DIR):

    # ===== 出力ディレクトリを毎回作り直す（既存解析は削除し更新） =====
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    d = df.copy()

    # 列名のチェック
    need = {"mDEratio", "mL", "mEN"}
    if not need <= set(d.columns):
        missing = need - set(d.columns)
        raise SystemExit(f"[ERROR] 必須列が不足: {missing}")

    # group が無ければダミー
    if "group" not in d.columns:
        d["group"] = "All"

    # ===== 4指標の読み込み（あれば） =====
    highlight_df = None
    if os.path.exists("results.csv"):
        try:
            h = pd.read_csv("results.csv")
            rename_map_h = {
                "DEratio": "mDEratio",
                "L":       "mL",
                "L_entr":  "mEN",
                "TT":      "mTT",
                "LAM":     "mLAM",
            }
            h = h.rename(columns={k: v for k, v in rename_map_h.items() if k in h.columns})
            h = h[h["name"].isin(INDICATOR_COLORS.keys())].copy()
            h["indicator"] = h["name"]
            h["short_label"] = h["indicator"].map(INDICATOR_SHORT)
            highlight_df = h
            print("Loaded 4 indicators from results.csv")
        except Exception as e:
            print("[WARN] failed to load results.csv for indicators:", e)

    # ========= 3D 散布図 =========
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

        # 4指標も3Dに重ねる
        if highlight_df is not None:
            for _, row in highlight_df.iterrows():
                name = row["indicator"]
                cind = INDICATOR_COLORS.get(name, "k")
                ax.scatter(float(row["mDEratio"]),
                           float(row["mL"]),
                           float(row["mEN"]),
                           c=[cind], s=180, marker="*",
                           edgecolor="black", linewidths=0.8,
                           label=name)
                ax.text(float(row["mDEratio"]),
                        float(row["mL"]),
                        float(row["mEN"]),
                        INDICATOR_SHORT.get(name, name),
                        fontsize=9, fontweight="bold")

        ax.set_xlabel("mDEratio")
        ax.set_ylabel("mL")
        ax.set_zlabel("mEN")
        ax.set_title("RQA Parameter Space (3D)")

        # 凡例（グループ＋4指標）
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h_, l_ in zip(handles, labels):
            uniq[l_] = h_
        ax.legend(uniq.values(), uniq.keys(), loc="best")

        f3d = os.path.join(out_root, "rqa_param_space_3d.png")
        fig.savefig(f3d, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {f3d}")
    except Exception as e:
        print("[WARN] 3D plot skipped:", e)

    # ========= 2D ペア（基本3つ） =========
    base_pairs = [("mDEratio", "mL"),
                  ("mDEratio", "mEN"),
                  ("mL", "mEN")]

    zoom_ranges_base = {
        ("mDEratio", "mL"):  {"xlim": (5, 24), "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    }

    # 全体
    out_full = os.path.join(out_root, "rqa_param_space_pairs.png")
    scatter_pairs(d, base_pairs,
                  out_path=out_full,
                  title="RQA Parameter Space (2D pairs)",
                  zoom_ranges=None,
                  annotate=True,
                  annotate_step=1,
                  highlight_df=highlight_df)

    # 拡大
    out_zoom = os.path.join(out_root, "rqa_param_space_pairs_zoom.png")
    scatter_pairs(d, base_pairs,
                  out_path=out_zoom,
                  title="RQA Parameter Space (2D pairs, ZOOM)",
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
                      title="RQA Parameter Space (TT pairs)",
                      zoom_ranges=None,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

        out_tt_zoom = os.path.join(out_root, "rqa_param_space_pairs_TT_zoom.png")
        scatter_pairs(d, tt_pairs,
                      out_path=out_tt_zoom,
                      title="RQA Parameter Space (TT pairs, ZOOM)",
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
                      title="RQA Parameter Space (LAM vs TT)",
                      zoom_ranges=None,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

        out_lam_zoom = os.path.join(out_root, "rqa_param_space_pairs_LAM_TT_zoom.png")
        scatter_pairs(d, lam_pairs,
                      out_path=out_lam_zoom,
                      title="RQA Parameter Space (LAM vs TT, ZOOM)",
                      zoom_ranges=zoom_lam,
                      annotate=True,
                      annotate_step=1,
                      highlight_df=highlight_df)

    # ===== 2D 用のポイントも CSV に保存 =====
    cols = ["file", "group", "rank",
            "mDEratio", "mL", "mEN", "mTT", "mLAM"]
    cols = [c for c in cols if c in d.columns]
    out_csv = os.path.join(out_root, "rqa_param_space_points.csv")
    d[cols].to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    df = pd.read_csv("rqa_summary_sort.csv")
    plot_rqa_param_space(df, out_root="4_grouping_result")
