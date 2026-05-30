import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """
    x:状態変数
    x[0]:変位r_x
    x[1]:変位r_y
    x[2]:速度v_x
    x[3]:速度v_y
    f:状態方程式
    """
    g = 9.8
    f = np.zeros(4)
    f[0] = x[2]
    f[1] = x[3]
    f[2] = 0
    f[3] = -g
    return f

def eular(f, x, h):
    """
    f:微分方程式f(x)
    x:状態変数
    h:オイラー法ステップ
    """
    df = f(x)
    y = np.zeros(4)
    y[0] = x[0] + h * df[0]
    y[1] = x[1] + h * df[1]
    y[2] = x[2] + h * df[2]
    y[3] = x[3] + h * df[3]
    return y

# main
h = 0.01
v0 = 20.0 # 初速度
theta = 45.0# 打ち上げ角度
rad = np.radians(theta)
v_0_x = v0 * np.cos(rad)
v_0_y = v0 * np.sin(rad)
x0 = np.array([0.0, 0.0, v_0_x, v_0_y])
t = 0.0
x_data = np.array([x0[0]])
y_data = np.array([x0[1]])
t_data = np.array([t])

# オイラー法で解軌道の計算
x = x0
while x[1] >= 0:
    x = eular(func, x, h)
    t += h
    x_data = np.append(x_data, x[0])
    y_data = np.append(y_data, x[1])
    t_data = np.append(t_data, t)

# グラフ描画
fig, ax = plt.subplots()
ax.set_ylabel("y(t)")
ax.set_xlabel("x(t)")
ax.plot(x_data, y_data, '-o')
plt.grid(True)
plt.show()
