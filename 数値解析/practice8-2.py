import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x

#RK法で1ステップずつ進める
def rk4(f, x, h):
    """
    f: 微分方程式 f(x)
    x: 状態変数
    h: ルンゲクッタ法ステップ
    """
    k1 = h * f(x)
    k2 = h * f(x + 0.5 * k1)
    k3 = h * f(x + 0.5 * k2)
    k4 = h * f(x + k3)
    y = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# main
h = 0.1
x0 = 1.0
t = 0.0
x_rk4 = x0
t_data = [t]
x_data_rk4 = [x_rk4]

while t <= 1.0:
    x_rk4 = rk4(func, x_rk4, h)
    t += h
    t_data.append(t)
    x_data_rk4.append(x_rk4)


#解析解
x_exact = [x0 * np.exp(tt) for tt in t_data]   

# グラフ描画
fig, ax = plt.subplots()
ax.set_ylabel("x(t)")
ax.set_xlabel("t")
ax.plot(t_data, x_data_rk4, '-o')
plt.plot(t_data, x_exact, '--', label='Exact')
plt.show()
