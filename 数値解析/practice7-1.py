import numpy as np
import matplotlib.pyplot as plt

#関数定義

#元の関数
def org_func(x):
    #sin(cosx)の微分: dy/du(du/dx)=dy/dx
    return np.sin(np.cos(x))
#1階微分11 
def diff_func(x): 
#sin(cosx)の微分：dy/du(du/dx)=dy/dx
 return -np.cos(np.cos(x)) * np.sin(x)

#前進差分商
def forward_diff(func, x, h):
    return (func(x + h) - func(x)) / h
    
#後退差分商
def backwards_diff(func, x, h):
    return (func(x) - func(x - h)) / h

    
#中心差分商
def central_diff(func, x, h):
    return (func(x + h) - func(x - h)) / (2 * h) 

#main
def main():
   h = 0.01
   x0 = np.pi / 4

   #数値微分
   fwd = forward_diff(org_func, x0, h)
   bwd = backwards_diff(org_func, x0, h)
   cen = central_diff(org_func, x0, h)

   #解析解
   true_val = diff_func(x0)

   print(f"x = pi/4 ≈ {x0:.5f}")
   print(f"True derivative: {true_val:.8f}")
   print(f"Forward difference : {fwd:.8f}")
   print(f"Backward difference : {bwd:.8f}")
   print(f"Central difference : {cen:.8f}")

   #グラフを出力
   X = np.linspace(0, np.pi, 200)
   Y = org_func(X)
   dY_analytical = diff_func(X)
   dY_numeriacal = central_diff(org_func, X, h)

   plt.figure(figsize=(10, 6))
   plt.plot(X, Y, label="f(x) = sin(cos(x))")
   plt.plot(X, dY_analytical, label="Analytical derivative")
   plt.plot(X, dY_numeriacal, '--', label="Central diffrence approx")
   plt.scatter([x0], [org_func(x0)], color='red', label="x=pi/4")
   plt.legend()
   plt.xlabel("x")
   plt.ylabel("y")
   plt.title("Function and Derivatives")
   plt.grid(True)
   plt.show() 
   
if __name__ == "__main__":
    main()

