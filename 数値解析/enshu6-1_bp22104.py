import numpy as np
import matplotlib.pyplot as plt
import sys

def least_squares_method(x, y, d=1):
    #近似したい点(x,y)、何時の多項式を作成するのか(デフォルトはd=1で行う)
    m = len(x)
    mat_a=np.zeros((2,2)) #p(x)=a+bxの係数を求めるための方程式

    
    b=np.zeros(2)

    #係数行列を構成
    mat_a[0][0] = m
    mat_a[0][1] = np.sum(x)
    mat_a[1][0] = np.sum(x)
    mat_a[1][1] = np.sum(x**2)

    #定数ベクトルを計算
    b[0]= np.sum(y)
    b[1]= np.sum(x * y)

    #求める係数(a,b)はmat_aを係数行列、bを定数ベクトルとして連立方程式を解いた解である
    #ガウスの消去法で解ベクトルを求める

    #前進消去法
    mat_a, b = forward_elimination(mat_a, b, 2)
    #後退消去法
    coeffs = backward_substitution(mat_a, b, 2)

    return coeffs
    
print('===Forward Elimination===')
def forward_elimination(mat_a,b,dim):
    for i in range(dim):
        # 部分ピボッティング、i行列目の最大値をもつ行を探す
        max_row = i + np.argmax(np.abs(mat_a[i:, i]))
        
        if np.abs(mat_a[max_row, i])<1e-8:
            #　ピボットが0になったらエラーで終了
            print("Pivot is zero\n")
            sys.exit(-1)
        #行の入れ替え
        if max_row != i:
            mat_a[[i, max_row]] = mat_a[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
        
        pivot = mat_a[i][i]

        print(f"Pivot element: {mat_a[i][i]}")

        for j in range(i+1,dim):
                # j = i 行からdim行まで前進消去を行う
                factor = mat_a[j][i] / pivot
                mat_a[j, i:] -= factor * mat_a[i, i:]
                b[j] -= factor * b[i]
        
        print("Matrix A after this step:")
        print(mat_a)
        print("vector b after this step:")
        print(b)
    

    return mat_a,b

def backward_substitution(mat_a,b,dim):
   x = np.zeros(dim) #解ベクトル
   print('===Backwards Eliminate===')
   for i in range(dim-1,-1,-1):
      # for文の意味: iをdim-1から0までデクリメントさせる
      if np.abs(mat_a[i][i]) < 1e-8:
          sys.exit(-1)
     
      x[i] = (b[i] - np.dot(mat_a[i, i+1:], x[i+1:])) / mat_a[i][i]
      print(f"x[{i}] = {x[i]}") 
   return x

def approx_linear_func(a,x):
    #y=a0+a1xを返す関数
    #a:多項式の係数ベクトル, x:多項式y=f(x)の変数
    return a[0] + a[1] * x

#main
x = np.array([0.5, 1.0, 1.5, 2.0])
y = np.array([1.65, 2.72, 4.48, 7.39])

#最小二乗法で近似
coeffs = least_squares_method(x, y)
print(f"Estimated coefficients: {coeffs}")

#グラフ描写
plt.scatter(x, y, label='Data')
x_vals = np.linspace(min(x), max(x), 100)
y_vals = approx_linear_func(coeffs, x_vals)
plt.plot(x_vals, y_vals, color='red', label='Fitted line')
plt.legend()
plt.title('Least Squares Liner Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
