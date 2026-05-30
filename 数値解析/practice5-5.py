import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# 関数 f(x) = 1 / (x^2 + 25)
def f(x):
    return 1.0 / (x**2 + 25)

# 補間点（等間隔）
x_nodes = np.linspace(-15, 15, 21)
y_nodes = f(x_nodes)

# ラグランジュ補間多項式
poly = lagrange(x_nodes, y_nodes)

# 描画用
x_fine = np.linspace(-15, 15, 1000)
y_true = f(x_fine)
y_interp = poly(x_fine)

plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_true, 'k-', label='f(x) = 1 / (x² + 25)')
plt.plot(x_fine, y_interp, 'r--', label='Lagrange Interpolation')
plt.plot(x_nodes, y_nodes, 'bo', label='Interpolation Points')
plt.ylim(-0.2, 1.2)
plt.title("Runge Phenomenon in Lagrange Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
