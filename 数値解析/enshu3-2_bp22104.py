import numpy as np
import sys


def inverse(mat_a):
    dim = mat_a.shape[0]
    #単位行列を作成
    mat_i = np.identity(dim)

    #A | Iの形に連結
    aug_mat = np.hstack((mat_a.astype(float), mat_i))

    for i in range(dim):
     #部分ピポッティング
     #列iの絶対値最大の要素を選んで、行を入れ替え
     max_row = i + np.argmax(np.abs(mat_a[i:, i])) 
          
     if np.abs(mat_a[max_row, i]) < 1e-8:
            #　ピボットが0になったらエラーで終了
            print("Pivot is zero. Matroix is singular\n")
            sys.exit(-1)
            
     if max_row != i:
         aug_mat[[i, max_row]] = aug_mat[[max_row, i]]

     #ピボットを1にする
     pivot = aug_mat[i][i]
     aug_mat[i] = aug_mat[i] / pivot

     #他の行を0にする
     for j in range(dim):
        if j != i:
            factor = aug_mat[j][i]
            aug_mat[j] -= factor * aug_mat[i]
    
#A^-1は右側に現れる
    inverse_mat = aug_mat[:, dim:]
    
    return inverse_mat
#main
#使用する行列式を定義
mat_a = np.asarray([[1.0, -4.0, -3.0],
                    [-1.0, 5.0, 3.0],
                    [-1.0, 6.0, 4.0]])

inv_A = inverse(mat_a)
print("Inverse A:")
print(inv_A)

#検証 (A * A^-1 = I)
print("A * A^-1 =")
print(np.dot(mat_a, inv_A))
        
 
