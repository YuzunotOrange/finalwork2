#ロジスティックス写像のカオス
#課題1-1
import numpy as np
import matplotlib.pyplot as plt


#ロジスティックスマップ関数
def logistic_map(a, x_0, iterate):
    x_n = x_0
    values = [x_n]
    for i in range(iterate):
        x_n = a*x_n*(1-x_n)
        values.append(x_n)
    return values
    
#初期値と試行回数の設定
x_0 = 0.101
iterate = 1000

#a=2.3の場合
a_2_3 = 2.3
values_2_3 = logistic_map(a_2_3, x_0, iterate)

#a=4.0の場合
a_4_0 = 4.0
values_4_0 = logistic_map(a_4_0, x_0, iterate)

#a=2.3のグラフ
plt.subplot(1, 2, 1)
plt.plot(values_2_3, color='blue')
plt.title("Logistic Map(a = 2.3)")
plt.xlabel("n")
plt.ylabel("x_n")

#a-4.0のグラフ
plt.subplot(1, 2, 2)
plt.plot(values_4_0, color='red')
plt.title("Logistic Map(a = 4.0)")
plt.xlabel("n")
plt.ylabel("x_n")

plt.tight_layout()
plt.show()


