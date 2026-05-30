import numpy as np
import matplotlib.pyplot as plt

#関数定義

#積分する関数（上半円の弧）
def circle_upper(x):
    return np.sqrt(1 - x**2)

#真値(解析解)
true_area = np.pi

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
a, b = -1, 1
n = 10

left_rect = 2 * peicewise_int(circle_upper, a, b, n)
trapezoid = 2 * trapezoidal_int(circle_upper, a, b, n)
simpson = 2 * simpson_int(circle_upper, a, b, n)

#結果表示
print(f"真値(解析積分): {true_area:.8f}")
print(f"区分求積法: {left_rect:.8f} 誤差: {abs(left_rect - true_area):.8f}")
print(f"台形法: {trapezoid:.8f} 誤差: {abs(trapezoid - true_area):.8f}")
print(f"シンプソン法: {simpson:.8f} 誤差: {abs(simpson - true_area):.8f}")

