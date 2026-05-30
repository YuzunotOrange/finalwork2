import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

#指定された関数
def f(x):
    return 1.0 / (x**2 +25)

#補間点を作成し、ラグランジュ補間で多項式を作る関数
def lagrange_interpolation(num_points):
    x_points = np.linspace(-15, 15, num_points)
    y_points = 1.0 / (x_points**2 + 25)
    
    poly = lagrange(x_points, y_points)
    return poly, x_points, y_points

#x軸点
x_fine = np.linspace(-15, 15, 1000)
y_true = f(x_fine)

#10点補間
poly_10, x_10, y_10 = lagrange_interpolation(10)
y_10 = poly_10(x_fine)
y_10_points = f(x_10)

#20点補間
poly_20, x_20, y_20 = lagrange_interpolation(20)
y_20 = poly_20(x_fine)
y_20_points = f(x_20)

#グラフ描画
plt.figure(figsize=(10,6))
plt.plot(x_fine, y_true, label="True function", color="black", linewidth = 2)
plt.plot(x_fine, y_10, label = "Lagrange Interpolation (10 points)", linestyle="--", color = "blue")
plt.plot(x_fine, y_20, label = "Lagrange Interpolation (20 points)", linestyle ="--", color = "red")
plt.scatter(x_10, y_10_points, color="blue", marker='o', s=40, label="interpolation points (10)")
plt.scatter(x_20, y_20_points, color= "red", marker='x', s=40, label="Interpolation points (20)")

plt.title("Runge Phenomenon with Lagrange Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()