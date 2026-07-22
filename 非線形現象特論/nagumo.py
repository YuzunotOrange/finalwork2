import numpy as np
import matplotlib.pyplot as plt

def rk4(t, x, f, h):
    """
    t : 時間
    x : 解ベクトル
    f : dx/dt = f(t,x)
    h : 刻み幅
    """
    k1 = f(t, x)

    xtemp = x + h/2.0 * k1
    k2 = f(t + h/2.0, xtemp)

    xtemp = x + h/2.0 * k2
    k3 = f(t + h/2.0, xtemp)

    xtemp = x + h * k3
    k4 = f(t + h, xtemp)

    return x + (k1 + 2.0*(k2 + k3) + k4) * h / 6.0


# パラメータ
a = 0.7
b = 0.8
c = 0.08


def solve(I):

    def func(t, x):
        V = x[0]
        u = x[1]

        dVdt = V - V**3/3 - u + I
        dudt = c * (V - b*u + a)

        return np.array([dVdt, dudt])

    # 初期条件
    x = np.array([-1.0, -0.5])

    T = 300.0
    h = 0.01
    N = int(T/h)

    t = 0.0

    Vhist = np.zeros(N+1)
    time = np.zeros(N+1)

    Vhist[0] = x[0]

    for i in range(N):
        x = rk4(t, x, func, h)
        t += h

        Vhist[i+1] = x[0]
        time[i+1] = t

    return time, Vhist


# I = 0.34
t1, V1 = solve(0.34)

# I = 0.23
t2, V2 = solve(0.23)

# グラフ
plt.figure(figsize=(10,5))
plt.plot(t1, V1, label='I = 0.34')
plt.xlabel('Time')
plt.ylabel('V')
plt.title('FitzHugh-Nagumo Model (I=0.34)')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(t2, V2, label='I = 0.23')
plt.xlabel('Time')
plt.ylabel('V')
plt.title('FitzHugh-Nagumo Model (I=0.23)')
plt.grid()
plt.legend()
plt.show()