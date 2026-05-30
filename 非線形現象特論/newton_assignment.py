import numpy as np
import matplotlib.pyplot as plt
import cmath

"""
f(z)=z^3-1=0をNewton法で解く．
解はz=1,ω,ω^2
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
    return complex(1000, 1000)

def f(x):
    """
    f(x)
    """
    return x**3-1

def df(x):
    """
    df(x)/dx
    """
    return 3*x**2

#探索範囲
"""
Re:[-1,1]
Im:[-1,1]
の範囲をそれぞれ512分割する
xxは実軸，yyは虚軸に対応
"""
xx=np.linspace(-1,1,512) #Real axsis
yy=np.linspace(-1,1,512) #Image axsis

roots = [complex(1,0), 
         complex(-0.5, np.sqrt(3)/(2)),
         complex(-0.5, -np.sqrt(3)/(2))
         ]

z = np.zeros((len(xx), len(yy)))

#ブルートフォース法による探索
for i in range(len(xx)):
    for j in range(len(yy)):
        res = newton(complex(xx[i], yy[j]), f, df)

        if abs(res - roots[0]) < 0.1:
            z[j][i] = 0
        elif abs(res - roots[1]) < 0.1:
            z[j][i] = 1
        elif abs(res - roots[2]) < 0.1:
            z[j][i] = 2
        else:
            z[j][i]
#Graph
fig = plt.figure()
ax=fig.add_subplot(111)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
ax.set_title('Newton Fractal: $z^3 - 1 = 0$')
img = ax.imshow(z,extent=(-1,1,-1,1),cmap='jet', origin='lower',interpolation='none',)
cbar = plt.colorbar(img, ticks=[-1, 0, 1, 2], label='Root Index')
plt.show()