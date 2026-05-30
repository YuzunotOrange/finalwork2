import scipy.interpolate as scipy
import numpy as np
import matplotlib.pyplot as plt

# 補間点
x = np.array([0.00, 0.50, 1.50, 2.00])
y = np.array([1.00, 1.65, 4.48, 7.39])

# ラグランジュ補間
l_poly = scipy.lagrange(x, y)

# 係数の出力（昇べき順に並べて表示、有効数字3桁）
print('coef =', np.round(l_poly.coef[::-1], 3))  # coef[::-1]で昇べき順

# 補間多項式のプロット（確認用）
xx = np.linspace(0, 2, 100)
yy = l_poly(xx)

plt.plot(xx, yy, label='Lagrange interpolation')
plt.plot(x, y, 'ro', label='Data points')
plt.xlabel('x')
plt.ylabel('p3(x)')
plt.title('Lagrange Polynomial Interpolation')
plt.legend()
plt.grid(True)
plt.show()
