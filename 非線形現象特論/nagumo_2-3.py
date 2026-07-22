import numpy as np
import matplotlib.pyplot as plt

def rk4(t, x, f, h):
    k1 = f(t, x)
    k2 = f(t + h/2.0, x + h/2.0 * k1)
    k3 = f(t + h/2.0, x + h/2.0 * k2)
    k4 = f(t + h, x + h * k3)

    return x + (k1 + 2.0*k2 + 2.0*k3 + k4) * h / 6.0


# パラメータ
a = 0.7
b = 0.8
c = 0.08


def solve(I, x0):
    def func(t, x):
        V = x[0]
        w = x[1]

        dVdt = V - V**3 / 3 - w + I
        dwdt = c * (V - b*w + a)

        return np.array([dVdt, dwdt])

    T = 300.0
    h = 0.01
    N = int(T / h)

    t = 0.0
    x = np.array(x0, dtype=float)

    time = np.zeros(N + 1)
    Vhist = np.zeros(N + 1)
    whist = np.zeros(N + 1)

    Vhist[0] = x[0]
    whist[0] = x[1]

    for i in range(N):
        x = rk4(t, x, func, h)
        t += h

        time[i+1] = t
        Vhist[i+1] = x[0]
        whist[i+1] = x[1]

    return time, Vhist, whist


# 初期値
x0 = [-1.0, -0.5]

# I = 0.48, 0.25
t1, V1, w1 = solve(0.48, x0)
t2, V2, w2 = solve(0.25, x0)


# V-w 相図
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(V1, w1)
plt.xlabel('V')
plt.ylabel('w')
plt.title('Phase Plane: I = 0.48')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(V2, w2)
plt.xlabel('V')
plt.ylabel('w')
plt.title('Phase Plane: I = 0.25')
plt.grid()

plt.tight_layout()
plt.show()

initial_values = [
    [-1.0, -0.5],
    [0.0, 0.0],
    [1.0, 1.0],
    [-2.0, 1.0]
]

plt.figure(figsize=(12,5))

# I = 0.48
plt.subplot(1,2,1)

for x0 in initial_values:
    t, V, w = solve(0.48, x0)
    plt.plot(V, w, label=f'{x0}')

plt.xlabel('V')
plt.ylabel('w')
plt.title('Phase Plane: I = 0.48')
plt.grid()
plt.legend()

# I = 0.25
plt.subplot(1,2,2)

for x0 in initial_values:
    t, V, w = solve(0.25, x0)
    plt.plot(V, w, label=f'{x0}')

plt.xlabel('V')
plt.ylabel('w')
plt.title('Phase Plane: I = 0.25')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()