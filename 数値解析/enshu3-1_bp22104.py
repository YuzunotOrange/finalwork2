import numpy as np
import sys
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
#係数行列と解ベクトルの定義
mat_a = np.array([[1.0, -2.0, 4.0],
                  [1.0, -1.0, 1.0],
                  [1.0, 0.0, 1.0]])
b = np.array([10.0, 21.0, 30.0])
dim = 3
mat_a2,b2=forward_elimination(mat_a.copy(),b.copy(),dim)
vec_x=backward_substitution(mat_a2,b2,dim)
print('===Final solution===')
print('vec_x =',[f"{v:.1f}"for v in vec_x])