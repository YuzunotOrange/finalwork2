import os
import glob
import numpy as np
from scipy import interpolate

# スクリプトのあるディレクトリ
target_dir = os.path.dirname(os.path.abspath(__file__))

# 出力フォルダ new_li を作成（存在しなければ）
out_dir = os.path.join(target_dir, "new_li")
os.makedirs(out_dir, exist_ok=True)

# ディレクトリ内のすべてのCSVファイルを取得
csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

if not csv_files:
    print("No CSV files found in directory.")
    exit(1)

for fname in csv_files:
    print(f"Processing: {fname}")

    # データ読み込み
    x_data = np.loadtxt(fname, usecols=[0], delimiter=',')
    y_data = np.loadtxt(fname, usecols=[1], delimiter=',')

    # 正規化と補間用データ生成
    x_data = (x_data - x_data[0]) / 7
    x_latent = np.arange(0, x_data[-1], 1)

    f_curve = interpolate.PchipInterpolator(x_data, y_data)

    # 出力ファイル名
    base = os.path.splitext(os.path.basename(fname))[0]
    outname = base + '_liner.csv'
    outpath = os.path.join(out_dir, outname)

    # CSV保存
    np.savetxt(outpath, f_curve(x_latent), delimiter=',')
    print(f" -> Saved: {outpath}")

print("All CSV files processed and saved in 'new_li' folder.")
