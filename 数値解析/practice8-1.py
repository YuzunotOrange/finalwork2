import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x

def euler(f, x, h):
    """
    f: 微分方程式 f(x)
    x: 状態変数
    h: オイラー法ステップ
    """
    y = x + h * f(x)
    return y

# main
h = 0.1
x0 = 1.0
t = 0.0

x_data = np.array([x0])
t_data = np.array([t])

while t <= 1.0:
    x0 = euler(func, x0, h)
    t += h
    x_data = np.append(x_data, x0)
    t_data = np.append(t_data, t)

# グラフ描画
fig, ax = plt.subplots()
ax.set_ylabel("x(t)")
ax.set_xlabel("t")
ax.plot(t_data, x_data, '-o')
plt.show()
