import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSVファイルのパス
file_path = "lorenz.csv"  # 実際のファイル名に置き換えてください

# データ読み込み
# ヘッダーなし、空白区切りの場合
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# 列名を付ける
data.columns = ['x', 'y', 'z']

# 3Dプロット
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# アトラクタ描画
ax.plot(data['x'], data['y'], data['z'], linewidth=0.7)

# 軸ラベル
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# タイトル
#ax.set_title('3D Attractor Visualization')

plt.show()
