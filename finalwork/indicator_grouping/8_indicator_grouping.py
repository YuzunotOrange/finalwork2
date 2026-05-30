#2025-12-15
#ローレンツアトラクタ、ロジスティックス写像、sin波、ホワイトノイズに関して指標を決めRQAしたもの
#追加で一次元ブラウン運動とローレンツアトラクタ、sine波、ロジスティックス写像それぞれにホワイトノイズを施した指標を追加
#指標それぞれに関する値を出力する
#指標それぞれに関して20%、50%、70%、90%の破損を加えて線形補間を施している
#実データと指標の破損率ごとの分布を重ね合わせることで比較できるようになっている


import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAIN_OUT_DIR = "8_grouping_with_Femilat_result"   # 出力ルート

# ===== 実データ側グループ色 =====
GROUP_COLORS = {
    "Low":  plt.cm.tab20(0),
    "Mid":  plt.cm.tab20(1),
    "High": plt.cm.tab20(2),
    "All":  plt.cm.tab20(3),
}
DEFAULT_COLOR = plt.cm.tab20(4)

# ===== 8指標の欠損率ごとのマーカー =====
MISSING_MARKERS = {0.0: "o", 0.2: "s", 0.5: "^", 0.7: "D", 0.9: "P"}
MISSING_LABELS  = {0.0: "0%", 0.2: "20%", 0.5: "50%", 0.7: "70%", 0.9: "90%"}

# ===== 指標色（ベース4つは固定、それ以外はtab20） =====
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

# ===== ZOOM 範囲（指標ごとに上書き可能） =====

ZOOM_DEFAULT = {
    "DE": {
        ("mDEratio", "mL"):  {"xlim": (5, 24),  "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    },
    "TT": {
        ("mDEratio", "mTT"): {"xlim": (10, 25), "ylim": (0, 45)},
        ("mL", "mTT"):       {"xlim": (0, 25),  "ylim": (0, 40)},
        ("mEN", "mTT"):      {"xlim": (0, 3),   "ylim": (0, 40)},
    },
    "LAM_TT": {
        ("mLAM", "mTT"): {"xlim": (0.6, 1.0), "ylim": (0, 50)},
    },
}

# 指標ごとの個別指定（必要なものだけ書けばOK）
ZOOM_BY_INDICATOR = {
    "Logistic map": {
        "DE": {
            ("mDEratio", "mL"): {"xlim": (10, 17), "ylim": (2.5, 7.5)},
            ("mDEratio", "mEN"): {"xlim": (10, 17), "ylim": (1.4, 2.2)},
            ("mL", "mEN"): {"xlim": (2.5, 7.5), "ylim": (1.4, 2.2)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (10, 17), "ylim": (2.5, 7.5)},
            ("mL", "mTT"): {"xlim": (2.5, 7.5), "ylim": (2.5, 7.5)},
            ("mEN", "mTT"): {"xlim": (1.4, 2.2), "ylim": (2.5, 7.5)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0, 0.8), "ylim": (2.5, 7.5)},
        },
    },
    "Logistic map + noise": {
        "DE": {
            ("mDEratio", "mL"): {"xlim": (7.5, 17.5), "ylim": (0, 10)},
            ("mDEratio", "mEN"): {"xlim": (7.5, 17.5), "ylim": (0, 2.5)},
            ("mL", "mEN"): {"xlim": (0, 10), "ylim": (0, 2.5)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (7.5, 17.5), "ylim": (0, 10)},
            ("mL", "mTT"): {"xlim": (0, 10), "ylim": (0, 10)},
            ("mEN", "mTT"): {"xlim": (0, 2.5), "ylim": (0, 10)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.3, 0.85), "ylim": (0, 10)},
        },
    },
    "Lorenz attractor (x)": {
        "DE": {
            ("mDEratio", "mL"): {"xlim": (18, 22), "ylim": (0, 20)},
            ("mDEratio", "mEN"): {"xlim": (18, 22), "ylim": (2.5, 3.5)},
            ("mL", "mEN"): {"xlim": (0, 20), "ylim": (2.5, 3.5)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (18, 22), "ylim": (5, 15)},
            ("mL", "mTT"): {"xlim": (7, 15), "ylim": (7, 15)},
            ("mEN", "mTT"): {"xlim": (2.5, 3.5), "ylim": (5, 15)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.95, 1.0), "ylim": (5, 15)},
        },
    },
    "Lorenz attractor (x) + noise":{
        "DE": {
            ("mDEratio", "mL"): {"xlim": (17.5, 20), "ylim": (5, 10)},
            ("mDEratio", "mEN"): {"xlim": (18, 20), "ylim": (2.5, 3.0)},
            ("mL", "mEN"): {"xlim": (6.0, 10.0), "ylim": (2.5, 3.0)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (18, 20), "ylim": (5, 15)},
            ("mL", "mTT"): {"xlim": (5, 10), "ylim": (7, 15)},
            ("mEN", "mTT"): {"xlim": (2.5, 3.0), "ylim": (5, 15)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.92, 1.0), "ylim": (5, 15)},
        },
    },
    "Sine wave":{
        "DE": {
            ("mDEratio", "mL"): {"xlim": (15, 25), "ylim": (20, 80)},
            ("mDEratio", "mEN"): {"xlim": (15, 25), "ylim": (3.0, 4.0)},
            ("mL", "mEN"): {"xlim": (20, 70), "ylim": (3.0, 4.0)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (19, 21), "ylim": (0, 10)},
            ("mL", "mTT"): {"xlim": (20, 70), "ylim": (0, 10)},
            ("mEN", "mTT"): {"xlim": (3.5, 4.0), "ylim": (0, 10)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.95, 1.5), "ylim": (0, 7)},
        },
    },
    "Sine wave + noise": {
        "DE": {
            ("mDEratio", "mL"): {"xlim": (8, 15.0), "ylim": (0, 10)},
            ("mDEratio", "mEN"): {"xlim": (8, 15.0), "ylim": (0, 2.0)},
            ("mL", "mEN"): {"xlim": (0, 10), "ylim": (0, 2.0)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (10, 14), "ylim": (0, 5)},
            ("mL", "mTT"): {"xlim": (0, 5), "ylim": (0, 5)},
            ("mEN", "mTT"): {"xlim": (0.5, 2.0), "ylim": (0, 5)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.4, 0.9), "ylim": (0, 5)},
        },
    },
    "White noise": {
        "DE": {
            ("mDEratio", "mL"): {"xlim": (8, 15.0), "ylim": (0, 10)},
            ("mDEratio", "mEN"): {"xlim": (8, 15.0), "ylim": (0.75, 2.5)},
            ("mL", "mEN"): {"xlim": (0, 10), "ylim": (0.75, 2.5)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (8, 15.0), "ylim": (2, 10)},
            ("mL", "mTT"): {"xlim": (0, 10), "ylim": (2, 10)},
            ("mEN", "mTT"): {"xlim": (0.75, 2.5), "ylim": (2, 10)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.4, 0.9), "ylim": (2, 10)},
        },
    },
    "Brownian motion":{
        "DE": {
            ("mDEratio", "mL"): {"xlim": (20, 23), "ylim": (0, 20)},
            ("mDEratio", "mEN"): {"xlim": (20, 23), "ylim": (2.0, 4.0)},
            ("mL", "mEN"): {"xlim": (0, 20), "ylim": (2.0, 4.0)}
        },
        "TT": {
            ("mDEratio", "mTT"): {"xlim": (20, 23), "ylim": (0, 20)},
            ("mL", "mTT"): {"xlim": (0, 20), "ylim": (0, 20)},
            ("mEN", "mTT"): {"xlim": (2.0, 4.0), "ylim": (0, 20)},
        },
        "LAM_TT": {
            ("mLAM", "mTT"): {"xlim": (0.9, 1.0), "ylim": (0, 20)},
        },
    },
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

def load_indicators_with_missing(csv_path="results_missing_compare.csv"):
    """
    results_missing_compare.csv を読み込み、列名を m*** に揃えて highlight_df を返す。
    想定列：
      - (DEratio, L, L_entr, TT, LAM) or (mDEratio, mL, mEN, mTT, mLAM)
      - indicator / base_name / name
      - missing_rate （0.0, 0.2, 0.5, 0.7, 0.9）
    """
    if not os.path.exists(csv_path):
        print(f"[INFO] indicator csv not found: {csv_path}")
        return None

    try:
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

        if "short_label" not in h.columns:
            h["short_label"] = h["indicator"].map(get_indicator_short)

        # missing_rate が "20%" や "0.2" 文字列の可能性もあるので正規化
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

            # 0/20/50/70/90%に絞る（混入対策）
            h = h[h["missing_rate"].isin([0.0, 0.2, 0.5, 0.7, 0.9])].copy()

        print(f"Loaded indicators: {csv_path}  (n={len(h)})")
        return h

    except Exception as e:
        print("[WARN] failed to load indicator csv:", e)
        return None

def plot_realdata_with_each_indicator_all_2d(
    df_real,
    highlight_df,
    out_root,
    annotate=True,
    annotate_step=1,
    # --- どの図を出すか ---
    make_de=True,
    make_tt=True,
    make_lam_tt=True,
):
    """
    指標ごとに、実データ + その指標（欠損率付き）を重ねた2D図をまとめて出力
      - DEペア:  (mDEratio–mL, mDEratio–mEN, mL–mEN)
      - TTペア:  (mDEratio–mTT, mL–mTT, mEN–mTT)
      - LAM–TT:  (mLAM–mTT)
    各セットで「全体図」と「ZOOM図」を出す。
    """

    if highlight_df is None or highlight_df.empty:
        print("[INFO] No indicator data. Skipped.")
        return

    base_dir = os.path.join(out_root, "real_vs_each_indicator")
    os.makedirs(base_dir, exist_ok=True)

    # ===== ペア定義 =====
    de_pairs = [("mDEratio", "mL"),
                ("mDEratio", "mEN"),
                ("mL", "mEN")]
    zoom_de = {
        ("mDEratio", "mL"):  {"xlim": (5, 24),  "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    }

    tt_pairs = [("mDEratio", "mTT"),
                ("mL", "mTT"),
                ("mEN", "mTT")]
    zoom_tt = {
        ("mDEratio", "mTT"): {"xlim": (10, 25), "ylim": (0, 45)},
        ("mL", "mTT"):       {"xlim": (0, 25),  "ylim": (0, 40)},
        ("mEN", "mTT"):      {"xlim": (0, 3),   "ylim": (0, 40)},
    }

    lam_pairs = [("mLAM", "mTT")]
    zoom_lam = {
        ("mLAM", "mTT"): {"xlim": (0.6, 1.0), "ylim": (0, 50)},
    }

    # ===== 指標ごとに出力 =====
    indicators = sorted(highlight_df["indicator"].unique())

    for ind in indicators:
                # --- ZOOM 範囲を指標ごとに取得 ---
        zoom_de = ZOOM_DEFAULT["DE"].copy()
        zoom_tt = ZOOM_DEFAULT["TT"].copy()
        zoom_lam = ZOOM_DEFAULT["LAM_TT"].copy()

        if ind in ZOOM_BY_INDICATOR:
            zind = ZOOM_BY_INDICATOR[ind]
            if "DE" in zind:
                zoom_de.update(zind["DE"])
            if "TT" in zind:
                zoom_tt.update(zind["TT"])
            if "LAM_TT" in zind:
                zoom_lam.update(zind["LAM_TT"])

        h_sub = highlight_df[highlight_df["indicator"] == ind].copy()
        if h_sub.empty:
            continue

        short = h_sub["short_label"].iloc[0] 
        if short in ZOOM_BY_INDICATOR:
            zind = ZOOM_BY_INDICATOR[short]
        out_dir = os.path.join(base_dir, f"indicator_{str(short).replace(' ', '_')}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Plot real data + indicator: {ind} ===")

        # --- DE ---
        if make_de:
            out_full = os.path.join(out_dir, f"real_vs_{short}_DE.png")
            scatter_pairs(df_real, de_pairs, out_full,
                          title=f"Real data vs {ind} (DE pairs)",
                          zoom_ranges=None,
                          annotate=annotate, annotate_step=annotate_step,
                          highlight_df=h_sub)

            out_zoom = os.path.join(out_dir, f"real_vs_{short}_DE_zoom.png")
            scatter_pairs(df_real, de_pairs, out_zoom,
                          title=f"Real data vs {ind} (DE pairs, ZOOM)",
                          zoom_ranges=zoom_de,
                          annotate=annotate, annotate_step=annotate_step,
                          highlight_df=h_sub)

        # --- TT ---
        if make_tt:
            # 実データ側に mTT がない場合はスキップ
            if "mTT" in df_real.columns:
                out_full = os.path.join(out_dir, f"real_vs_{short}_TT.png")
                scatter_pairs(df_real, tt_pairs, out_full,
                              title=f"Real data vs {ind} (TT pairs)",
                              zoom_ranges=None,
                              annotate=annotate, annotate_step=annotate_step,
                              highlight_df=h_sub)

                out_zoom = os.path.join(out_dir, f"real_vs_{short}_TT_zoom.png")
                scatter_pairs(df_real, tt_pairs, out_zoom,
                              title=f"Real data vs {ind} (TT pairs, ZOOM)",
                              zoom_ranges=zoom_tt,
                              annotate=annotate, annotate_step=annotate_step,
                              highlight_df=h_sub)
            else:
                print("[INFO] Real data has no mTT column -> TT pairs skipped.")

        # --- LAM vs TT（連続性） ---
        if make_lam_tt:
            # 実データ側に mLAM と mTT がない場合はスキップ
            if ("mLAM" in df_real.columns) and ("mTT" in df_real.columns):
                out_full = os.path.join(out_dir, f"real_vs_{short}_LAM_TT.png")
                scatter_pairs(df_real, lam_pairs, out_full,
                              title=f"Real data vs {ind} (LAM vs TT)",
                              zoom_ranges=None,
                              annotate=annotate, annotate_step=annotate_step,
                              highlight_df=h_sub)

                out_zoom = os.path.join(out_dir, f"real_vs_{short}_LAM_TT_zoom.png")
                scatter_pairs(df_real, lam_pairs, out_zoom,
                              title=f"Real data vs {ind} (LAM vs TT, ZOOM)",
                              zoom_ranges=zoom_lam,
                              annotate=annotate, annotate_step=annotate_step,
                              highlight_df=h_sub)
            else:
                print("[INFO] Real data has no mLAM/mTT -> LAM vs TT skipped.")



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
    実データ散布図（グループ色）に、highlight_df（指標）を重ね描きする
    - zoom_ranges があれば、その範囲に入る点だけ描画
    - highlight_df は missing_rate があればマーカーを変える
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_scale*n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    d = d.copy()

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

    # 指標一覧（色決定のため）
    all_inds = None
    if highlight_df is not None and "indicator" in highlight_df.columns:
        all_inds = set(highlight_df["indicator"].unique())

    # 指標凡例の重複防止（この図全体で）
    used_indicator_legend = set()

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

        # ---- 実データ（グループごと） ----
        for g in np.unique(groups_vis):
            gm = (groups_vis == g)
            ax.scatter(x_vis[gm], y_vis[gm],
                       c=colors_vis[gm],
                       s=24, alpha=0.8,
                       edgecolors="none",
                       label=str(g))

        # ---- 実データ点の番号 ----
        if annotate:
            step = max(1, int(annotate_step))
            for i in range(0, len(x_vis), step):
                ax.text(float(x_vis[i]), float(y_vis[i]),
                        str(labels_vis[i]),
                        fontsize=7, ha="center", va="bottom",
                        alpha=0.7, clip_on=True)

        # ---- 指標（欠損率マーカー） ----
        if highlight_df is not None and (xcol in highlight_df.columns) and (ycol in highlight_df.columns):
            h = highlight_df.copy()

            if is_zoom:
                h = h[(h[xcol] >= xmin) & (h[xcol] <= xmax) &
                      (h[ycol] >= ymin) & (h[ycol] <= ymax)]

            for _, row in h.iterrows():
                name = row["indicator"]
                xh = float(row[xcol])
                yh = float(row[ycol])

                cind = get_indicator_color(name, all_inds)

                mr = row["missing_rate"] if "missing_rate" in row else None
                if mr is None or pd.isna(mr):
                    marker = "*"
                    ms_label = ""
                else:
                    mr = float(mr)
                    marker = MISSING_MARKERS.get(mr, "*")
                    ms_label = MISSING_LABELS.get(mr, f"{int(mr*100)}%")

                short = row.get("short_label", name)

                # 凡例：最初の軸で一度だけ（指標×欠損率）
                label = None
                if ax_idx == 0:
                    key = ms_label if ms_label else None
                    if key not in used_indicator_legend:
                        label = key
                        used_indicator_legend.add(key)

                ax.scatter(xh, yh,
                           c=[cind], s=160, marker=marker,
                           edgecolor="black", linewidths=0.8,
                           label=label, zorder=3)

                text = short if not ms_label else f"{ms_label}"
                if text:    
                    ax.text(xh, yh, text,
                        fontsize=9, fontweight="bold",
                        ha="left", va="bottom",
                        clip_on=True, zorder=4)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, ls=":", alpha=0.4)

    # ---- 凡例（グループ＋指標） ----
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    axes[-1].legend(uniq.values(), uniq.keys(), title="Group / Indicator", loc="best")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_rqa_param_space(df, out_root=MAIN_OUT_DIR, indicator_csv="results_missing_compare.csv"):
    # ===== 出力ディレクトリ初期化 =====
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    d = df.copy()

    # 必須列
    need = {"mDEratio", "mL", "mEN"}
    if not need <= set(d.columns):
        raise SystemExit(f"[ERROR] 必須列が不足: {need - set(d.columns)}")

    if "group" not in d.columns:
        d["group"] = "All"

    # ===== 指標読み込み =====
    highlight_df = load_indicators_with_missing(indicator_csv)

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

        # 指標は 3D は混雑しやすいので「0%のみ」表示（推奨）
        if highlight_df is not None and {"mDEratio", "mL", "mEN"} <= set(highlight_df.columns):
            h3d = highlight_df.copy()
            if "missing_rate" in h3d.columns:
                h3d = h3d[h3d["missing_rate"] == 0.0]

            all_inds = set(h3d["indicator"].unique()) if len(h3d) else None
            for _, row in h3d.iterrows():
                name = row["indicator"]
                cind = get_indicator_color(name, all_inds)
                ax.scatter(float(row["mDEratio"]),
                           float(row["mL"]),
                           float(row["mEN"]),
                           c=[cind], s=180, marker="*",
                           edgecolor="black", linewidths=0.8,
                           label=f"{name} (0%)")
                ax.text(float(row["mDEratio"]),
                        float(row["mL"]),
                        float(row["mEN"]),
                        row.get("short_label", name),
                        fontsize=9, fontweight="bold")

        ax.set_xlabel("mDEratio")
        ax.set_ylabel("mL")
        ax.set_zlabel("mEN")
        ax.set_title("RQA Parameter Space (3D)")

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
        ("mDEratio", "mL"):  {"xlim": (5, 24),  "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    }

    out_full = os.path.join(out_root, "rqa_param_space_pairs.png")
    scatter_pairs(d, base_pairs, out_full,
                  title="RQA Parameter Space (2D pairs)",
                  zoom_ranges=None,
                  annotate=True, annotate_step=1,
                  highlight_df=highlight_df)

    out_zoom = os.path.join(out_root, "rqa_param_space_pairs_zoom.png")
    scatter_pairs(d, base_pairs, out_zoom,
                  title="RQA Parameter Space (2D pairs, ZOOM)",
                  zoom_ranges=zoom_ranges_base,
                  annotate=True, annotate_step=1,
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
        scatter_pairs(d, tt_pairs, out_tt,
                      title="RQA Parameter Space (TT pairs)",
                      zoom_ranges=None,
                      annotate=True, annotate_step=1,
                      highlight_df=highlight_df)

        out_tt_zoom = os.path.join(out_root, "rqa_param_space_pairs_TT_zoom.png")
        scatter_pairs(d, tt_pairs, out_tt_zoom,
                      title="RQA Parameter Space (TT pairs, ZOOM)",
                      zoom_ranges=zoom_tt,
                      annotate=True, annotate_step=1,
                      highlight_df=highlight_df)

    # ========= LAM vs TT =========
    if {"mLAM", "mTT"} <= set(d.columns):
        lam_pairs = [("mLAM", "mTT")]
        zoom_lam = {
            ("mLAM", "mTT"): {"xlim": (0.6, 1.0), "ylim": (0, 50)},
        }

        out_lam = os.path.join(out_root, "rqa_param_space_pairs_LAM_TT.png")
        scatter_pairs(d, lam_pairs, out_lam,
                      title="RQA Parameter Space (LAM vs TT)",
                      zoom_ranges=None,
                      annotate=True, annotate_step=1,
                      highlight_df=highlight_df)

        out_lam_zoom = os.path.join(out_root, "rqa_param_space_pairs_LAM_TT_zoom.png")
        scatter_pairs(d, lam_pairs, out_lam_zoom,
                      title="RQA Parameter Space (LAM vs TT, ZOOM)",
                      zoom_ranges=zoom_lam,
                      annotate=True, annotate_step=1,
                      highlight_df=highlight_df)

    # ===== 実データ点のCSV保存 =====
    cols = ["file", "group", "rank", "mDEratio", "mL", "mEN", "mTT", "mLAM"]
    cols = [c for c in cols if c in d.columns]
    out_csv = os.path.join(out_root, "rqa_param_space_points.csv")
    d[cols].to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    # ===== 実データ読み込み =====
    df = pd.read_csv("rqa_summary_sort.csv")

    # ===== 指標（8指標・欠損率付き）読み込み → highlight_df を定義 =====
    highlight_df = load_indicators_with_missing("results_missing_compare.csv")

    # ===== 指標ごとに「実データ＋指標」を重ねた2D図を出力 =====
    pairs_2d = [
        ("mDEratio", "mL"),
        ("mDEratio", "mEN"),
        ("mL", "mEN"),
    ]

    zoom_ranges_2d = {
        ("mDEratio", "mL"):  {"xlim": (5, 24),  "ylim": (0, 50)},
        ("mDEratio", "mEN"): {"xlim": (10, 25), "ylim": (1.5, 4)},
        ("mL", "mEN"):       {"xlim": (0, 20),  "ylim": (1, 3.5)},
    }

    # 出力先フォルダを作り直す（必要なら）
    if os.path.exists(MAIN_OUT_DIR):
        shutil.rmtree(MAIN_OUT_DIR)
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)

    plot_realdata_with_each_indicator_all_2d(
    df_real=df,
    highlight_df=highlight_df,
    out_root=MAIN_OUT_DIR,
    annotate=True,
    annotate_step=1,
    make_de=True,
    make_tt=True,
    make_lam_tt=True
)

