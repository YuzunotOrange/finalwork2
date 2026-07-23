#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import time

# ===== 欠損と補間 =====
def introduce_block_missing_and_interpolate(x, missing_rate=0.0, block_len=None, seed=None):
    """
    連続ブロック欠損をランダムに付与し、線形補間で埋める。
    missing_rate : 全体の何割を欠損させるか (0 <= r < 1)
    block_len    : 1ブロック欠損長（サンプル数）。Noneなら自動推定。
    seed         : 乱数シード。Noneなら現在時刻から生成。
    戻り値:
      filled : 補間後の1次元配列
      mask   : True(=欠損させた点) / False のブーリアン配列
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("入力データが空です。")
    if not (0.0 <= missing_rate < 1.0):
        raise ValueError("missing_rate は 0 <= r < 1 の範囲で指定してください。")
    if missing_rate <= 0.0:
        return x.copy(), np.zeros(n, dtype=bool)

    total_missing = int(round(missing_rate * n))
    if block_len is None:
        block_len = max(1, total_missing // 10)

    if seed is None:
        seed = int(time.time()) & 0xffffffff

    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)

    num_blocks = max(1, total_missing // block_len)
    for _ in range(num_blocks):
        start = rng.integers(0, max(1, n - block_len + 1))
        mask[start:start + block_len] = True

    # 上限調整
    if mask.sum() > total_missing:
        extra = mask.sum() - total_missing
        idx_true = np.where(mask)[0]
        keep = rng.choice(idx_true, size=mask.sum() - extra, replace=False)
        mask[:] = False
        mask[keep] = True

    y = x.copy()
    y[mask] = np.nan

    # 線形補間（端点は最近傍値で補完）
    xi = np.arange(n)
    valid = ~np.isnan(y)
    if valid.sum() == 0:
        raise ValueError("全点が欠損になりました。missing_rate と block_len を見直してください。")
    filled = np.interp(xi, xi[valid], y[valid])

    return filled, mask

def main():
    ap = argparse.ArgumentParser(
        description="Make a 3-column file (time flux error) for the plotting script, after block-missing + linear interpolation."
    )
    ap.add_argument("file", help="入力データ（空白/CSV可）。1列 or 3列(time flux error) を想定")
    ap.add_argument("--missing-rate", type=float, required=True, help="欠損率 (0〜1)")
    ap.add_argument("--missing-block", type=int, default=None, help="1ブロック欠損長（省略時は自動推定）")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時は現在時刻ベースで自動）")
    ap.add_argument("--out", default=None, help="出力ファイル（省略時は <入力名>_filled.txt）")
    args = ap.parse_args()

    # 入力ファイルを頑健に読む（空白/CSV 自動判別）
    try:
        df = pd.read_csv(args.file, header=None, comment="#", sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"入力の読み込みに失敗しました: {e}")

    ncols = df.shape[1]
    if ncols < 1:
        raise ValueError("少なくとも1列の数値データが必要です。")

    # time/flux/error の取り出しロジック
    # - 3列以上: 0列目= time, 1列目= flux, 2列目= error（以降は無視）
    # - 1列のみ: 0列目を flux とみなし、time はインデックス、error は 0 にする
    if ncols >= 3:
        time_col = df.iloc[:, 0].to_numpy(dtype=float)
        flux_col = df.iloc[:, 1].to_numpy(dtype=float)
        err_col  = df.iloc[:, 2].to_numpy(dtype=float)
    elif ncols == 1:
        flux_col = df.iloc[:, 0].to_numpy(dtype=float)
        time_col = np.arange(len(flux_col), dtype=float)  # インデックスを time に
        err_col  = np.zeros_like(flux_col, dtype=float)
    else:
        # 2列（time, flux）の場合は error を 0 にする
        time_col = df.iloc[:, 0].to_numpy(dtype=float)
        flux_col = df.iloc[:, 1].to_numpy(dtype=float)
        err_col  = np.zeros_like(flux_col, dtype=float)

    # 欠損付与＋補間は flux のみ対象
    filled_flux, _ = introduce_block_missing_and_interpolate(
        flux_col,
        missing_rate=args.missing_rate,
        block_len=args.missing_block,
        seed=args.seed
    )

    # 出力（ヘッダなし、空白区切り、プロッタがそのまま読める形式）
    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = args.out if args.out is not None else f"{base}_filled.csv"

    out_arr = np.column_stack([time_col, filled_flux, err_col])
    # 空白区切り・ヘッダ無し・行番号無し・適度な浮動小数点フォーマット
    np.savetxt(out_path, out_arr, fmt="%.10g")

    print(f"[INFO] input_cols={ncols}, missing_rate={args.missing_rate}, block_len={args.missing_block}, seed={args.seed}")
    print(f"[OK] Wrote 3-column file for your plotter: {out_path}")
    print("       (columns = time flux error ; whitespace-separated, no header)")

if __name__ == "__main__":
    main()
