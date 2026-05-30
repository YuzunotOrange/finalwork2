import numpy as np
import matplotlib.pyplot as plt

#関数定義

#元の関数
def org_func(x):
    #sin(cosx)の微分: dy/du(du/dx)=dy/dx
    return x**(-2)

#積分 (真値を計算) 
def int_func(a,b):
    return (1/a) - (1/b) 

#区分求積法 
def peicewise_int(func,a,b,n):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    y = func(x)
    return h * np.sum(y)  

#func:元の関数,積分区間[a,b], n:分割数，ステップは(b-a)/n 
#台形法 
def trapezoidal_int(func,a,b,n):  
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

#func:元の関数,積分区間[a,b], n:分割数，ステップは(b-a)/n 
#シンプソン法
def simpson_int(func,a,b,n):  
#n:シンプソン法では偶数になることに注意！
# func:元の関数,積分区間[a,b], n:分割数，ステップは(b-a)/n 
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's rule")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return h * S / 3


#main 
a, b = 1, 2
n = 10
true_value = int_func(a, b)

left_rect = peicewise_int(org_func, a, b, n)
trapezoid = trapezoidal_int(org_func, a, b, n)
simpson = simpson_int(org_func, a, b, n)

#結果表示
print(f"真値(解析積分): {true_value:.8f}")
print(f"区分求積法: {left_rect:.8f} 誤差: {abs(left_rect - true_value):.8f}")
print(f"台形法: {trapezoid:.8f} 誤差: {abs(trapezoid - true_value):.8f}")
print(f"シンプソン法: {simpson:.8f} 誤差: {abs(simpson - true_value):.8f}")

#グラフ描画
x_vals = np.linspace(a, b, 200)
y_vals = org_func(x_vals)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="f(x) = x^-2")
plt.title("f(x) = x^-2 on [1, 2]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show() 