#ロジスティックス写像のカオス
#課題1-4
#疑似乱数の発生 平面へのプロット
import numpy as np
import matplotlib.pyplot as plt

#疑似乱数を1000個生成
random_values = np.random.rand(1000)

#データ生成
x_n = random_values[:-1]
x_n_plus_1 = random_values[1:]

#プロット
plt.figure(figsize=(8, 6))
plt.plot(x_n, x_n_plus_1, linestyle='-', marker= '.', markersize=3)
plt.title("$x_n$ vs $x_{n+1}$ Plot of Pseudo-Random Numbers")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+1}$")
plt.grid(True)
plt.show()