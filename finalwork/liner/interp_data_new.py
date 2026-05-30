#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import interp1d

def load_two_columns_csv(path: Path, delimiter: str = ","):
    """
    CSVを2列（x,y）として読み込む。
    先頭行にヘッダがある場合でもなるべく吸収する。
    """
    try:
        data = np.loadtxt(path, delimiter=delimiter, usecols=[0, 1])
    except ValueError:
        # ヘッダ付きの可能性を考慮して genfromtxt で再トライ
        data = np.genfromtxt(path, delimiter=delimiter)
        if data.ndim == 1 and data.size >= 2:
            data = data.reshape(-1, 2)
        if data.shape[1] < 2:
            raise ValueError(f"{path}: 2列の数値データとして読み込めませんでした。")
        data = data[:, :2]
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    return x, y

def process_one_file(csv_path: Path, out_dir: Path, show_progress: bool = True):
    # 入力読み込み
    x_data, y_data = load_two_columns_csv(csv_path, delimiter=",")

    # 元コードの前処理と同じ：x を (x - x0) / 7 に正規化
    x_norm = (x_data - x_data[0]) / 7.0
    # 整数刻みのレイテンシ軸（0 から x_norm の末尾まで、ステップ 1）
    x_latent = np.arange(0, x_norm[-1], 1)

    if len(np.unique(x_norm)) < 2:
        raise ValueError(f"{csv_path}: x が定数のため補間できません。")

    # ---- 線形補間（interp1d(kind='linear')）----
    # 範囲外は外挿しない（＝評価しない）前提なので、x_latent は範囲内にしてあります
    f_linear = interp1d(x_norm, y_data, kind="linear", bounds_error=True)

    y_interp = f_linear(x_latent)

    # 出力パス
    base = csv_path.stem  # 拡張子を除いたファイル名
    outname = base + "_liner.csv"
    outpath = out_dir / outname

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outpath, y_interp, delimiter=",")
    if show_progress:
        print(f"[OK] {csv_path.name} -> {outpath}")

def main():
    parser = argparse.ArgumentParser(
        description="lc フォルダ内のCSVすべてに線形補間を適用して *_liner.csv を出力します。"
    )
    parser.add_argument(
        "base_dir",
        help="対象となる lc フォルダを含む“ベースディレクトリ”のパス（例: /path/to/project）",
    )
    parser.add_argument(
        "--lc-name",
        default="lc",
        help="lc フォルダ名（既定: lc）"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="出力ディレクトリ（既定: <base_dir>/<lc-name>/processed）"
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="対象CSVのグロブ（既定: *.csv）"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    lc_dir = base_dir / args.lc_name

    if not lc_dir.is_dir():
        print(f"Error: lc ディレクトリが見つかりません: {lc_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (lc_dir / "processed")

    csv_files = sorted(lc_dir.glob(args.glob))
    if not csv_files:
        print(f"Warning: 対象CSVが見つかりませんでした: {lc_dir} / pattern={args.glob}", file=sys.stderr)
        sys.exit(0)

    print(f"Processing directory: {lc_dir}")
    print(f"Output directory    : {out_dir}")
    print(f"Files found         : {len(csv_files)}")

    # 逐次処理
    n_ok, n_err = 0, 0
    for csv_path in csv_files:
        try:
            process_one_file(csv_path, out_dir, show_progress=True)
            n_ok += 1
        except Exception as e:
            print(f"[ERR] {csv_path.name}: {e}", file=sys.stderr)
            n_err += 1

    print(f"Done. OK={n_ok}, ERR={n_err}")

if __name__ == "__main__":
    main()
