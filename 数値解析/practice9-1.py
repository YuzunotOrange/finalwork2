import numpy as np
import matplotlib.pyplot as plt

g = 9.8
v0 = 9.8

def func(x):
    """
    x:状態変数
    x[0]:変位
    x[1]:速度
    f:状態方程式
    """
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = -g
    return f

def rk4(f, x, h):
    ############
    # f:微分方程式f(x)
    # x:状態変数
    # h:ルンゲクッタ法ステップ
    ###########
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 =f(x + h * k3)
    y = x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def main():
    # main
    # 初期化
    h = 0.1
    #状態変数の初期化
    x = np.array([0.0, v0])
    t = 0.0
    x_data = [x[0]]
    t_data = [t]
    # 解軌道の計算
    #時間発展
    t_max = 2.0
    while t < t_max:
        x = rk4(func, x, h)
        t += h
        if x[0] < 0:
            break
        x_data = np.append(x_data, x[0])
        t_data = np.append(t_data, t)
    # グラフ描画
    plt.plot(t_data, x_data, marker = 'o')
    plt.title('Vertical Throw Upwards (Runge-Kutta Method)')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.show()
   
if __name__ == "__main__":
    main()
