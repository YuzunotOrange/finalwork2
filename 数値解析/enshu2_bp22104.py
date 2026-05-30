import numpy as np
#次元数を入力
print("正方行列の次元数（dim）を入力してください\n")
dim_a = int(input("a:"))

#空のリスト
matrix_a = []

#成分を1個づつ入力
for i in range(dim_a):
    row_a = []
    for j in range(dim_a):
        value = float(input(f"行{i+1}、列{j+1}の成分を入力してください: "))
        row_a.append(value)
    matrix_a.append(row_a)

#NumPy配列に変換
matrix_a = np.array(matrix_a)

#bの行列を作成
print("正方行列の次元数（dim）を入力してください\n")
dim_b = int(input("b:"))

#空のリスト
matrix_b = []

#成分を1個づつ入力
for i in range(dim_b):
    row_b = []
    for j in range(dim_b):
        value = float(input(f"行{i+1}、列{j+1}の成分を入力してください: "))
        row_b.append(value)
    matrix_b.append(row_b)

#NumPy配列に変換
matrix_b = np.array(matrix_b)

#掛け算
matrix_c = np.matmul(matrix_a, matrix_b)

print("matrix_c=", matrix_c)

