#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    num_blocks = max(1, total_missing // block_len)

    for _ in range(num_blocks):
        start = rng.integers(0, n - block_len + 1)
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

# ===== メイン =====
def main():
    ap = argparse.ArgumentParser(description="Recurrence Plot with block missing and interpolation (m=1, tau=1)")
    ap.add_argument("file", help="入力データ（テキスト/CSV, 空白区切りOK, 1列目使用）")
    ap.add_argument("--missing-rate", type=float, default=0.0, help="欠損率 (0〜1, default=0)")
    ap.add_argument("--missing-block", type=int, default=10, help="1ブロック欠損長 (default=10)")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード (default=0)")
    ap.add_argument("--radius", type=float, default=None, help="RP半径（未指定時はデータ標準偏差の20%）")
    args = ap.parse_args()

    # データ読み込み（空白区切り1列目）
    data = np.loadtxt(args.file, usecols=[0])
    base = os.path.splitext(os.path.basename(args.file))[0]

    # 欠損＆補間
    data_filled, mask = introduce_block_missing_and_interpolate(
        data, missing_rate=args.missing_rate, block_len=args.missing_block, seed=args.seed
    )

    # 半径
    radius = args.radius if args.radius is not None else 0.2 * np.std(data_filled)

    print(f"[INFO] m=1, tau=1, radius={radius:.6g}, missing_rate={args.missing_rate}, block_len={args.missing_block}")

    # PyRQA設定
    ts = TimeSeries(data_filled, embedding_dimension=1, time_delay=1)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(radius),
        similarity_measure=EuclideanMetric
    )

    # RP生成
    rp = RPComputation.create(settings).run()
    M = np.array(rp.recurrence_matrix)

    # 描画
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(M, cmap="binary", origin="lower", interpolation="none")
    plt.title(f"Recurrence Plot (m=1, tau=1)\nMissing {args.missing_rate*100:.1f}%")
    plt.xlabel("Time"); plt.ylabel("Time")
    plt.tight_layout()
    png_path = f"{base}_m1_tau1_rp.png"
    plt.savefig(png_path)
    plt.close()
    print(f"[OK] RP saved: {png_path}")

if __name__ == "__main__":
    main()
