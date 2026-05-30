import numpy as np
#次元数を入力
print("ベクトルの次元数（dim）を入力してください\n")
dim_a = int(input("a:"))

#空のリスト
elements_a = []

#成分を1個づつ入力
for i in range(dim_a):
    value = float(input(f"{i+1}番目の成分を入力してください: "))
    elements_a.append(value)

#NumPy配列に変換
vec_a = np.array(elements_a)

#bのベクトルを作成
print("ベクトルの次元数（dim）を入力してください\n")
dim_b = int(input("b:"))

#空のリスト
elements_b = []

#成分を1個づつ入力
for i in range(dim_b):
    value = float(input(f"{i+1}番目の成分を入力してください: "))
    elements_b.append(value)

#NumPy配列に変換
vec_b = np.array(elements_b)

#スカラー積
vec_α = 10 * vec_a

#加減算
vec_c = vec_a + vec_b
vec_d = vec_a - vec_b

#内積
dot_ab = np.dot(vec_a, vec_b)

print("vec_a=", vec_a)
print("vec_b=", vec_b)
print("vec_α=", vec_α)
print("vec_c=", vec_c)
print("vec_d=", vec_d)
print("dot_ab=", dot_ab)