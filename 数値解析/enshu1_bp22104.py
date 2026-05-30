import numpy as np
import math
def quadratic_floma(a, b, c):
#a,b,c　を引数にしてax^2 + bx + c= 0の解を返す関数
    x=np.zeros(2)
    D = b**2-4*a*c #二次方程式の解の判別式　D=b^2-4ac
    if D>0:
        """
        if 判別式：
        　判別式が真の場合
        else :
        　判別式が偽の場合
        """
        sqrt_D = np.sqrt(D)
        if b >= 0:
            x_1 = (-b - sqrt_D) / (2*a)
        else:
            x_1 = (-b + sqrt_D) / (2*a) #ここの計算で桁落ちの可能性
        x_2 = c / (a * x_1) #解と係数の関係により桁落ちを防ぐ
        x[0] = x_1
        x[1] = x_2
        return x
        
    
    else:
        #実数解がない場合はNaNを返す
        return np.array([math.nan, math.nan])
#main
a=3.0
b=-350
c=2
x=quadratic_floma(a,b,c)
print(f'{x[0]:.15e},{x[1]:.15e}')
