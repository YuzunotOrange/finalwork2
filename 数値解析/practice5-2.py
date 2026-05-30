import scipy.interpolate as scipy
import numpy as np

#x,y：補間点
x=np.array([-2.0,-1.0,0.0])
y=np.array([-3.0,2.0,1.0])

l_poly=scipy.lagrange(x,y)

print('coef=',l_poly.coef[::-1])
#l_poly.coef：ラグランジュ補間により求めた多項式の係数を返す」

start , stop = -2, 0
xx = np.linspace(start,stop,10)
yy = l_poly(xx)