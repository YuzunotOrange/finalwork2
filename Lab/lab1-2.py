#ロジスティックス写像のカオス
#課題1-2
import numpy as np
import matplotlib.pyplot as plt

#初期値と試行回数の設定
a = 4.0
x_0 = 0.101
iterate = 1000

#ロジスティックスマップ関数
def logistic_map(a, x_0, iterate):
    x_n = x_0
    values = [x_n]
    for i in range(iterate):
        x_n = a*x_n*(1-x_n)
        values.append(x_n)
    return values
    
#データ生成
values = logistic_map(a, x_0, iterate)
x_n = values[:-1]
x_n_plus_1 = values[1:]

#プロット
plt.figure(figsize=(6,6))
plt.scatter(x_n, x_n_plus_1, s=1, alpha=0.5, color='blue')
plt.title("Logstic Map (a = 4.0) in $x_n$ vs $x_{n+1}$ plane")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+1}$")
plt.grid(True)
plt.show()