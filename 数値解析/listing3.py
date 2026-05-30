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
        x[0] = ((-b+np.sqrt(D))/2*a)
        x[1] = ((-b-np.sqrt(D))/2*a)
        return x
    
    else:
        #実数解がない場合はNaNを返す
        return math.nan
#main
a=3.0
b=-350
c=2
x=quadratic_floma(a,b,c)
print(f'{x[0]:.15e},{x[1]:.15e}')
