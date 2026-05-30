import numpy as np
import matplotlib.pyplot as plt

#練習問題２の内容を使用
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

#練習問題1の内容を使用
def rk4(f, x, h):
    """
    ルンゲクッタ4次法で状態を1ステップ進める
    """
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    y = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4) 
    return y

#練習問題2の内容を使用
# main
h = 0.01
v0 = 5.0  # 初速度
for theta in [np.pi / 4, np.pi / 8]:
    v_0_x = v0 * np.cos(theta)
    v_0_y = v0 * np.sin(theta)
    x = np.array([0.0, 0.0, v_0_x, v_0_y])
    t = 0.0

    x_data = [x[0]]
    y_data = [x[1]]
    t_data = [t]

    # ルンゲクッタ法で軌道を計算
    while True:
        next_x = rk4(func, x, h)
        if next_x[1] < 0:
            break
        x = next_x
        t += h
        x_data.append(x[0])
        y_data.append(x[1])
        t_data.append(t)
    
    angel_deg = int(np.degrees(theta))
    print(f"For θ = π/{int(np.pi/theta)} (approx. {angel_deg}°):")
    print(f" Time until landing: {t:.2f} s")
    print(f" Horizontal ditance: {x[0]:.2f} m\n")

    plt.plot(x_data, y_data, label=f"θ = π/{int(np.pi/theta)}")

# グラフ描画
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Projectile Motion by RK4")
plt.grid(True)
plt.legend()
plt.show()
