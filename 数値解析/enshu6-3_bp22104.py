import numpy as np
import matplotlib.pyplot as plt
import sys

def least_squares_method(x, y, d=1):
    #近似したい点(x,y)、何時の多項式を作成するのか(デフォルトはd=1で行う)
    dim = d + 1
    mat_a=np.zeros((dim,dim)) #p(x)=a+bxの係数を求めるための方程式
    b = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            mat_a[i][j] = np.sum(x **(i + j))
        #定数ベクトルを計算
        b[i] = np.sum(y * x ** i)


    #求める係数(a,b)はmat_aを係数行列、bを定数ベクトルとして連立方程式を解いた解である
    #ガウスの消去法で解ベクトルを求める

    #前進消去法
    mat_a, b = forward_elimination(mat_a, b, dim)
    #後退消去法
    coeffs = backward_substitution(mat_a, b, dim)
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

    #多項式関数(1次、2次)
def approx_linear_func(a,x):
    #y=a0+a1xを返す関数
    #a:多項式の係数ベクトル, x:多項式y=f(x)の変数
    return a[0] + a[1] * x

def approx_quadratic_func(a, x):
    #y=a0+a1x+a2x^2を返す関数
    #a:多項式の係数ベクトル, x:多項式y=f(x)の変数
        return a[0] + a[1] * x + a[2] * x ** 2



#main
x = np.array([0.5, 1.0, 1.5, 2.0])
y = np.array([1.65, 2.72, 4.48, 7.39])

#最小二乗法で近似
coeffs_liner = least_squares_method(x, y, d = 1)
coffes_quadratics = least_squares_method(x, y, d=2)

print(f"Liner coefficients: {coeffs_liner}")
print(f"Quadratic coefficients: {coffes_quadratics}")

#グラフ描写
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color = 'black', label='Data points')
x_vals = np.linspace(min(x), max(x), 200)
y_linear = approx_linear_func(coeffs_liner, x_vals)
y_quadratics = approx_quadratic_func(coffes_quadratics, x_vals)

plt.plot(x_vals, y_linear, color='red', label='Linear fit')
plt.plot(x_vals, y_quadratics, color='blue', label='Quadratic fit')

plt.title('Least Squares Approximation: Linear vs Quadratics')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
