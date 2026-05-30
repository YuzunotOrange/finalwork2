import numpy as np
import matplotlib.pyplot as plt

# 時系列データ（sin波）
t = np.linspace(0, 4 * np.pi, 300)
x = np.sin(t)

# リカレンス行列の計算
eps = 0.1  # 閾値（再帰の判定用）
N = len(x)
R = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if abs(x[i] - x[j]) < eps:
            R[i, j] = 1

# リカレンスプロットの表示
plt.imshow(R, origin='lower', cmap='binary')
plt.title('Recurrence Plot of sin(t)')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.show()
