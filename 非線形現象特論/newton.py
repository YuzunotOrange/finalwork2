import numpy as np
import matplotlib.pyplot as plt
import cmath

"""
f(z)=z^2-1=0をNewton法で解く．
解はz=1,-1
"""

def newton(x,f,df,eps=1.0e-8,maxiter=20):
    """
    x:初期値x_0
    f,df:関数f(x)，一階微分df(x)
    eps:許容誤差
    maxiter:最大試行回数（収束しなければ発散と見做す）
    """
    y=x
    for i in range(maxiter):
        y=y-f(y)/df(y)
        if (abs(f(y))<eps):
            #epsよりもf(y)が小さくなれば解としてyを返す
            return y
    print ('divergence')
    #収束しなかった場合は異常値として1000を返す
    return (1000)

def f(x):
    """
    f(x)
    """
    return x**2-1

def df(x):
    """
    df(x)/dx
    """
    return 2*x

#探索範囲
"""
Re:[-1,1]
Im:[-1,1]
の範囲をそれぞれ512分割する
xxは実軸，yyは虚軸に対応
"""
xx=np.linspace(-1,1,512) #Real axsis
yy=np.linspace(-1,1,512) #Image axsis

#絨毯爆撃
z=np.zeros((len(xx),len(yy)))
for i in range(len(xx)):
    for j in range(len(yy)):
        """
        複素数
        complex(x,y)=x+jy
        """
        omega=newton(complex(xx[i],yy[j]),f,df)

        """
        解の色分け:z=1,-1
        """
        #print(i,j,omega.real)
        z[j][i]=omega.real
#print(z)


#Graph
fig = plt.figure()
ax=fig.add_subplot(111)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
img = ax.imshow(z,extent=(-1,1,-1,1),cmap='jet', origin='lower',interpolation='none',)
cbar = plt.colorbar(img, ax=ax)
plt.show()