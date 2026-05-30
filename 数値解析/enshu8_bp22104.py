import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x

#オイラー法
def euler(f, x, h):
    """
    f: 微分方程式 f(x)
    x: 状態変数
    h: オイラー法ステップ
    """
    y = x + h * f(x)
    return y


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

#初期値
x_euler = x0
x_rk4 = x0

#データ保存用
t_data = [t]
x_data_euler = [x_euler]
x_data_rk4 = [x_rk4]


while t <= 1.0:
    x_euler = euler(func, x_euler, h)
    x_rk4 = rk4(func, x_rk4, h)
    t += h
    t_data.append(t)
    x_data_euler.append(x_euler)
    x_data_rk4.append(x_rk4)

#解析解
x_exact = [x0 * np.exp(tt) for tt in t_data]    

# グラフ描画
plt.figure()
plt.plot(t_data, x_data_euler, '-o', label='Euler')
plt.plot(t_data, x_data_rk4, '-s', label='RK4')
plt.plot(t_data, x_exact, '--', label='Exact')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Euler vs RK4')
plt.legend()
plt.show()
