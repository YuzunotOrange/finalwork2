import numpy as np

def f(x):
    #関数f(x)=x^2-2x-1を定義
    return x**2-2*x-1

def df(x):
    #f(x)の導関数df(x)/dxを定義
    return 2*x-2

def newton(eps,x0):

    while True:
        #while(1)は無限ループ
        #ニュートン法の反復計算
     x = x0 - (f(x0)/df(x0))
     if np.abs(x-x0) < eps:
        # np.abs(a) : aの絶対値を返す関数
        break
     else:
        x0 = x
    return x
    
#main
eps=1e-5
initial_guesses = [-1, 3]
solutions = np.zeros(2)
for i in range(len(initial_guesses)):
   solutions[i] = newton(eps, initial_guesses[i])

print(f'{solutions[0]:.5e},{solutions[1]:.5e}')
