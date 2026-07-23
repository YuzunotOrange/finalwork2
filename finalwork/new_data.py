#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Light curve plotter (no interpolation)")
    parser.add_argument("csv_file", help="空白区切り（またはCSV）で time flux error の3列が入ったファイル")
    args = parser.parse_args()

    # ファイル読み込み（コメント行は # でスキップ）
    # 空白区切りを想定。CSV かもしれない場合は sep=None, engine='python' で自動推定でもOK
    try:
        df = pd.read_csv(
            args.csv_file,
            header=None,
            delim_whitespace=True,   # CSV なら sep=',' に変更 or sep=None, engine='python'
            comment="#",
            on_bad_lines="skip"      # 古いpandasでは使えない場合あり
        )
    except TypeError:
        # 古い pandas 向けフォールバック（on_bad_lines が未対応な場合）
        df = pd.read_csv(
            args.csv_file,
            header=None,
            delim_whitespace=True,
            comment="#"
        )

    if df.shape[1] < 3:
        raise ValueError("入力ファイルには少なくとも 3 列（time, flux, error）が必要です。")

    # 列を分離（DataFrame は iloc/loc を使う）
    flux = df.iloc[:, 1].to_numpy(dtype=float)

    # インデックスを横軸
    index = np.arange(len(flux))

    # 点でプロット（補間なし）
    plt.figure()
    plt.plot(index, flux, '-', label='Observed Data')  # color指定は任意
    plt.xlabel("Time [weeks]")
    plt.ylabel("Flux")
    plt.title("Original Light Curve (No Interpolation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

