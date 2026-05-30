import numpy as np
import matplotlib.pyplot as plt

#関数定義

#元の関数
def org_func(x):
    #sin(cosx)の微分: dy/du(du/dx)=dy/dx
    return np.sin(np.cos(x))


#練習7-1から前進差分法と後退差分法を組み合わせた式
def central_diff_2nd(func, x, h):
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)


#main
def main():
   h = 0.01
   X_full = np.arange(0, 2*np.pi + h, h)
   f_full = org_func(X_full)
   
   #xの内側の点（差分が定義できる点）
   X_inner = X_full[1:-1]

   #2階微分
   d2Y_numeriacal = central_diff_2nd(org_func, X_inner, h)



   #グラフ
   plt.figure(figsize=(12, 6))
   plt.plot(X_full, f_full, label="f(x) = sin(cos(x))")
   plt.plot(X_inner, d2Y_numeriacal, '--', label="Central diffrence approx (2nd derivative)")
   plt.xlabel("x")
   plt.ylabel("y")
   plt.title("Function and Numerical 2nd Derivative (Centered Difference)")
   plt.legend()
   plt.grid(True)
   plt.show() 
   
if __name__ == "__main__":
    main()


