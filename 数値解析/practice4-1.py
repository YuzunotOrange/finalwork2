import numpy as np

def f(x):
    # f(x) = x^2-11
    return x**2-11

def bisection_method(eps,a,b):
    # eps:許容誤差, a,b:探索範囲a<bとする
    while(1):
        #探索範囲の中点を求める 
        m = (a + b) / 2
        #f(m)の符号を元に、探索範囲を更新する
        if f(m) * f(a) < 0:
            
            b = m 
        
        elif f(m) * f(b)> 0:

            a = m

        #終了条件はm<eps
        if abs(b - a) < eps:
            break

        return m
    
#main
eps = 1e-5
a= 3 #√11 = 3.31662
b= 4
x = bisection_method(eps,a,b)
print(f'{x:.5e}')