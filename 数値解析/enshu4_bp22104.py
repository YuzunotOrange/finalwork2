import numpy as np

def f(x):
    #関数f(x)=x^2-5を定義
    return x**2 - 5
    
def df(x):
    #f(x)の導関数df(x)/dxを定義
    return 2*x

#ニュートン法
def newton(eps, x0):
    x = x0 - f(x0) / df(x0)
    #ニュートン法の反復計算
    while True:
        x = x0 - f(x0) / df(x0)
        if abs(x - x0) < eps:
           # np.abs(a) : aの絶対値を返す関数
          break
        x0 = x
    return x

#main
eps = 1e-5 #10進数6桁精度
x0 = 2.0 #√5 ≈ 2.236 初期チ
root_approx = newton(eps, x0)

#少数6桁で出力
print(f'{root_approx:.5f}') 