#2026-1-8
#Bhattaとの図を比較するためのプログラム
#データ自体を種類で分けていない為、色付けの部分で問題アリ
#Figure3 まで記述が完了していてFIgure 4は明日以降取り組み予定

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 設定
csv_path = Path("rqa_summary.csv")
out_dir = Path("comapre_with_Bhatt")
out_dir.mkdir(parents=True, exist_ok=True)

y_col_fig4 = "mTT"

label_every = 1

#--1) 読み込み
df = pd.read_csv(csv_path)

df["source"] = (
    df["file"]
    .astype(str)
    .str.replace("_lc_linear.csv", "", regex=False)
    .str.replace(".csv", "", regex=False)
)

if "class" not in df.columns:
    df["class"] = "Unknown"

color_map = {"FSRQ": "red", "BL Lac": "blue", "Unknown": "gray"}
df["color"] = df["class"].map(color_map).fillna("gray")

df_fig = df.sort_values("mD", ascending=False).reset_index(drop=True)
df_fig["Index"] = df_fig.index + 1

mapping_cols = ["Index", "source", "file", "class", "mD", "mL", "mEN"]
if y_col_fig4 in df_fig.columns:
    mapping_cols.append(y_col_fig4)

mapping_path = out_dir / "index_mapping.csv"
df_fig[mapping_cols].to_csv(mapping_path, index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

panels = [
    ("mD", "meanD (mD)", "Figure3_left_meanD_vs_Index.png"),
    ("mL", "meanL (mL)", "Figure3_middle_meanL_vs_Index.png"),
    ("mEN", "meanENTR (mEN)", "Figure3_left_meanENTR_vs_Index.png")
]

for ax, (col, ylabel, fname) in zip(axes, panels):
    ax.scatter(df_fig["Index"], df_fig[col], c=df_fig["color"], s=35)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)

    for i, row in df_fig.iterrows():
        if (row["Index"] - 1) % label_every !=0:
            continue
        ax.annotate(
            str(int(row["Index"])),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize= 8
        )

fig.subtitle("FIgure 3 style (mD/mL/mEN vs Index)", y=1.02)
plt.tight_layout()

fig3_path = out_dir / "Figure3_3panels.png"
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)