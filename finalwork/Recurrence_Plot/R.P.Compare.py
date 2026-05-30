#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2つのリカレンスプロット（破損なし / 破損＋線形補間）を作成し、
差分プロットを出力するスクリプト。

依存: numpy, matplotlib, pyrqa
pip install numpy matplotlib pyrqa
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RPComputation
import matplotlib as mpl

mpl.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize": 20,   # 軸ラベルサイズ
    "axes.titlesize": 18,   # タイトル
})



# ===== 欠損と補間 =====
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
    num_blocks = max(1, max(0, total_missing) // max(1, block_len))

    for _ in range(num_blocks):
        start = rng.integers(0, max(1, n - block_len + 1))
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


# ===== RPの計算と保存 =====
def compute_rp_matrix(series, radius, m=1, tau=1):
    ts = TimeSeries(series, embedding_dimension=m, time_delay=tau)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(radius),
        similarity_measure=EuclideanMetric
    )
    rp = RPComputation.create(settings).run()
    M = np.array(rp.recurrence_matrix, dtype=np.uint8)  # 0/1
    return M


def save_rp_image(M, title, path_png):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(M, cmap="binary", origin="lower", interpolation="none")
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

def save_time_series_plot(y, title, path_png, xlabel="Time [weeks]", ylabel="Flux"):
    """
    破損なし時系列（元データ）を画像として保存する。
    横軸はサンプル番号（0..N-1）を使用。
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))

    plt.figure(figsize=(6.4, 4.8), dpi=150)
    plt.plot(x, y, linewidth=1.2, label="Observed Data")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


# ===== メイン =====
def main():
    ap = argparse.ArgumentParser(
        description="RP: 破損なし/破損＋補間 と 差分プロット（m=1, tau=1）"
    )
    ap.add_argument("file", help="入力データ（テキスト/CSV, 空白区切りOK, 1列目使用）")
    ap.add_argument("--missing-rate", type=float, default=0.2, help="欠損率 (0〜1, default=0.2)")
    ap.add_argument("--missing-block", type=int, default=20, help="1ブロック欠損長 (default=20)")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード (default=0)")
    ap.add_argument("--radius", type=float, default=None,
                    help="RP半径（未指定時は『元データ』標準偏差の20%）")
    ap.add_argument("--outdir", default=".", help="画像とnpzの出力先 (default=.)")
    ap.add_argument("--show", action="store_true", help="図を画面表示（保存も実施）")
    args = ap.parse_args()

    # --- データ読み込み ---
    data = np.loadtxt(args.file, usecols=[0])
    base = os.path.splitext(os.path.basename(args.file))[0]
    os.makedirs(args.outdir, exist_ok=True)

    # --- 半径（比較の公平性のため元データで算出） ---
    radius = args.radius if args.radius is not None else 0.2 * np.std(data)
    print(f"[INFO] m=1, tau=1, radius={radius:.6g}, "
          f"missing_rate={args.missing_rate}, block_len={args.missing_block}")

    # --- 破損なし RP ---
    M_clean = compute_rp_matrix(data, radius, m=1, tau=1)
    png_clean = os.path.join(args.outdir, f"{base}_clean_rp.png")
    save_rp_image(M_clean, "Recurrence Plot (m=1, tau=1)\nNo Missing", png_clean)
    print(f"[OK] RP (clean) saved: {png_clean}")

    # --- 破損＋補間データ & RP ---
    data_filled, miss_mask = introduce_block_missing_and_interpolate(
        data, missing_rate=args.missing_rate, block_len=args.missing_block, seed=args.seed
    )
    M_interp = compute_rp_matrix(data_filled, radius, m=1, tau=1)
    png_interp = os.path.join(args.outdir, f"{base}_interp_rp.png")
    title_interp = f"Recurrence Plot (m=1, tau=1)\nMissing {args.missing_rate*100:.1f}% (linear interp.)"
    save_rp_image(M_interp, title_interp, png_interp)
    print(f"[OK] RP (interpolated) saved: {png_interp}")

        # --- 破損なし時系列（元データ）プロットを保存 ---
    png_ts = os.path.join(args.outdir, f"{base}_clean_timeseries.png")
    save_time_series_plot(
        data,
        title="Original Light Curve (No Interpolation)",
        path_png=png_ts,
        xlabel="Time [weeks]",   # 必要なら "Time (index)" に変更OK
        ylabel="Flux"
    )
    print(f"[OK] Time series (clean) saved: {png_ts}")


    # --- 差分プロット（RP同士の絶対差：0=同じ, 1=異なる）---
    diff = np.abs(M_clean.astype(np.int16) - M_interp.astype(np.int16)).astype(np.uint8)
    diff_rate = diff.mean()  # 0〜1で、異なる画素の割合
    png_diff = os.path.join(args.outdir, f"{base}_rp_diff.png")
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(diff, cmap="gray", origin="lower", interpolation="nearest")
    plt.title(f"RP Difference (|clean - interp|)\nchanged={diff_rate*100:.2f}% of pixels")
    plt.xlabel("Time"); plt.ylabel("Time")
    plt.tight_layout()
    plt.savefig(png_diff)
    plt.close()
    print(f"[OK] Diff saved: {png_diff}")
    print(f"[STATS] changed pixels: {diff.sum()} / {diff.size} "
          f"({diff_rate*100:.2f}%)")

    # --- 便利: すべて保存（再解析用）---
    npz_path = os.path.join(args.outdir, f"{base}_rp_all.npz")
    np.savez_compressed(
        npz_path,
        data=data,
        data_filled=data_filled,
        missing_mask=miss_mask,
        M_clean=M_clean,
        M_interp=M_interp,
        diff=diff,
        radius=radius
    )
    print(f"[OK] NPZ saved: {npz_path}")

    # --- 任意表示 ---
    if args.show:
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1); plt.imshow(M_clean, cmap="binary", origin="lower", interpolation="none")
        plt.title("Clean"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(M_interp, cmap="binary", origin="lower", interpolation="none")
        plt.title("Interpolated"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(diff, cmap="gray", origin="lower", interpolation="nearest")
        plt.title("Difference"); plt.axis("off")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
