import numpy as np
import sys

def forward_elimination(mat_a,b,dim):
    for i in range(dim):
        # 部分ピポッティング、i行列目の最大値を持つ行を探す
        max_row = i + np.argmax(np.abs(mat_a[i:, i])) 

        if np.abs(mat_a[max_row, i]) < 1e-8:
            #　ピボットが0になったらエラーで終了
            print("Pivot is zero\n")
            sys.exit(-1)
        #行の入れ替え
        if max_row != i:
            mat_a[[i, max_row]] = mat_a[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]

        pivot = mat_a[i][i]
        
        for j in range(i+1,dim):
                # j = i 行からdim行まで前進消去を行う
                factor = mat_a[j][i] / pivot
                mat_a[j, i:] -= factor * mat_a[i, i:]
                b[j] -= factor * b[i]

    return mat_a,b

def backward_substitution(mat_a,b,dim):
   x = np.zeros(dim) #解ベクトル
   for i in range(dim-1,-1,-1):
      # for文の意味: iをdim-1から0までデクリメントさせる
      if np.abs(mat_a[i][i]) < 1e-8:
          sys.exit(-1)
     
      x[i] = (b[i] - np.dot(mat_a[i, i+1:], x[i+1:])) / mat_a[i][i]

   return x
#main
#係数行列と解ベクトルの定義
mat_a = np.array([[2.0, 1.0, -1.0],
                  [-3.0, -1.0, 2.0],
                  [-2.0, 1.0, 2.0]])
b = np.array([8.0, -11.0, -3.0])
dim = 3
mat_a2,b2=forward_elimination(mat_a.copy(),b.copy(),dim)
vec_x=backward_substitution(mat_a2,b2,dim)

print('vec_x =',vec_x)