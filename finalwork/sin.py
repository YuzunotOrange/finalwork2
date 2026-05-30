import numpy as np
import matplotlib.pyplot as plt

# 時間軸と sin 関数の値を生成
t = np.linspace(0, 4 * np.pi, 300)  # 時間（0 から 4π まで）
x = np.sin(t)                       # sin 関数の値

#インデックスを生成
timeindex = np.arange(len(x))

# グラフの描画
plt.plot(timeindex, x)
plt.title('Time Series of sin(t)')
plt.xlabel('Time Index')
plt.ylabel('sin(t)')
plt.grid(True)
plt.show()
