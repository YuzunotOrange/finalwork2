import numpy as np

#n次元ベクトル
vec_a=np.array([1,2,3])

#3行3列の行列
mat_a=np.array([[1,2,3],[2,2,3],[3,4,5]])

#単位行列
unit_mat=np.eye(3)

#零行列
zero_mat=np.zeros((3,3))

#空ベクトル
empty_vec=np.empty(3)

#空行列
empty_mat=np.empty((3,4))

print("vec_a=", vec_a)
print("mat_a=", mat_a)
print("unit_mat=", unit_mat)
print("zero_mat=", zero_mat)
