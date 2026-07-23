import os
import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # ついでに軸ラベルも大きくしたいなら（任意）
    "axes.labelsize": 20,
    "axes.titlesize": 20,
})



# スクリプトのあるディレクトリ
target_dir = os.path.dirname(os.path.abspath(__file__))

# 出力フォルダ new_li_2d を作成（存在しなければ）
out_dir = os.path.join(target_dir, "new_li_2d")
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
    x_data = (x_data - x_data[0]) / 7.0
    x_latent = np.arange(0, x_data[-1], 1.0)

    # 線形補間関数を作成
    f_curve = interpolate.interp1d(x_data, y_data, kind='linear')
    y_interp = f_curve(x_latent)

    # 出力ファイル名（PNG画像）
    base = os.path.splitext(os.path.basename(fname))[0]
    outname = base + '_liner.png'
    outpath = os.path.join(out_dir, outname)

    # グラフ描画
    fig = plt.figure()
    plt.plot(x_latent, y_interp, "-o", color="red", label="Interpolated (linear)")
    plt.plot(x_data, y_data, "o", color="blue", label="Actual data")
    plt.xlabel('Time [weeks]')
    plt.ylabel('Flux')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

    print(f" -> Saved plot: {outpath}")

print("All CSV files processed and plots saved in 'new_li' folder.")
