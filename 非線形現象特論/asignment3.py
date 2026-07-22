#非線形現象特論　
#課題３
#BP22014 松本　優瑞
import numpy as np
import matplotlib.pyplot as plt


a = 0.7

def dv(v, u):
    return v - v**3 - u


def du(v, u):
    return v + a


def rk4(t, x, f, h):
    
    k1 = f(t, x)

    xtemp = x + h / 2.0 * k1
    k2 = f(t, xtemp)

    xtemp = x + h / 2.0 * k2
    k3 = f(t, xtemp)

    xtemp = x + h * k3
    k4 = f(t, xtemp)

    return x + (k1 + 2.0 * (k2 + k3) + k4) * h / 6.0


def func(t, x):
    f = np.zeros(2)

    f[0] = dv(x[0], x[1])
    f[1] = du(x[0], x[1])

    return f


v0 = 2.0
u0 = 0.0



t = np.arange(0.0, 1000.0, 0.01)
h = 0.01


y0 = np.array([v0, u0], dtype=float)

v_vec = [v0]
u_vec = [u0]


for i in range(len(t)):
    y0 = rk4(t[i], y0, func, h)

    v_vec.append(y0[0])
    u_vec.append(y0[1])


plt.figure(figsize=(8, 7))


plt.plot(
    v_vec,
    u_vec,
    label="trajectory"
)


plt.plot(
    v_vec[0],
    u_vec[0],
    "o",
    color="orange",
    label="initial point"
)


# 平衡点
v_eq = -a
u_eq = a**3 - a

plt.plot(
    v_eq,
    u_eq,
    "ko",
    markersize=8,
    label="equilibrium point"
)


plt.xlabel("V")
plt.ylabel("u")
plt.title(f"Phase plane: a = {a}")


padding = 0.5

vmax = max(v_vec) + padding
vmin = min(v_vec) - padding

umax = max(u_vec) + padding
umin = min(u_vec) - padding



V, U = np.meshgrid(
    np.arange(vmin, vmax, 0.2),
    np.arange(umin, umax, 0.2)
)

dV = dv(V, U)
dU = du(V, U)

Vec = np.sqrt(dU**2 + dV**2)



Vec_safe = np.where(Vec == 0, 1.0, Vec)


plt.quiver(
    V,
    U,
    dV / Vec_safe,
    dU / Vec_safe,
    Vec,
    cmap="jet"
)

plt.colorbar(label="vector magnitude")


# ヌルクライン
V2, U2 = np.meshgrid(
    np.arange(vmin, vmax, 0.1),
    np.arange(umin, umax, 0.1)
)

dV2 = dv(V2, U2)
dU2 = du(V2, U2)


# dv/dt = 0
plt.contour(
    V2,
    U2,
    dV2,
    levels=[0],
    colors="green"
)


# du/dt = 0
plt.contour(
    V2,
    U2,
    dU2,
    levels=[0],
    colors="red"
)


plt.xlim([vmin, vmax])
plt.ylim([umin, umax])

plt.legend()
plt.grid()

plt.show()