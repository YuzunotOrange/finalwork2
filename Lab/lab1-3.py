#ロジスティックス写像のカオス
#課題1-3
#疑似乱数の発生
import numpy as np
import matplotlib.pyplot as plt

#疑似乱数を1000個生成
random_values = np.random.rand(1000)

#試行回数(0 ~ 999)
n = np.arange(1000)

#プロット
plt.figure(figsize=(10, 5))
plt.plot(n, random_values, linestyle='-', marker='.', markersize=3)
plt.title("Time Series Graph of Pseudo-Random Numbers")
plt.xlabel("Iteration count(n)")
plt.ylabel("Value of random numbers")
plt.grid(True)
plt.show()

