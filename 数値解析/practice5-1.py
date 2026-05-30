import numpy as np
import matplotlib.pyplot as plt
import sys

def polynomial(x,a,n):
    #係数ベクトルa,次数n-1多項式
    p=0
    for i in range(n): #(n-1)だと二次項が無視されてしまう
        p += a[i]*np.power(x,i)
    return p

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

#main
#補完点
x=np.array([-2.0,-1.0,0.0])
y=np.array([-3.0,2.0,1.0])
n = len(x) #次数 = n - 1 = 2

#バンデルモンド行列の生成
mat_a = np.vander(x, increasing=True) #[[1, x, x^2], ....]
b = y.copy()

#ガウスの消去法で解く
mat_a2,b2 = forward_elimination(mat_a.copy(), b.copy(), n)
a = backward_substitution(mat_a2, b2, n) #係数ベクトル

print("補完多項式の係数 a =", a)

#補完多項式
xx= np.linspace(-2,0,10)
yy= polynomial(xx,a,n)

fig,ax=plt.subplots()
ax.plot(x,y, "o",color='red')
ax.plot(xx,yy, '-',color='blue')
ax.set_title("Quadratic Interpolation Polynomial")
ax.legend()
ax.grid(True)
plt.show()