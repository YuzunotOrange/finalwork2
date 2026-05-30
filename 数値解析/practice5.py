import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(0,2*np.pi,20)
#np.linspace (start,stop,n) startからstopまでをn分割した数列を作成する
x = np.sin(t)
x1 = x+1
x2 = x-1

fig,ax = plt.subplots()
#Figureオブジェクト(図全体の描画領域)とAxesオブジェクトする座標系)を指定する
ax.set_xlabel('t')
ax.set_ylabel('x')
#軸のラベルを設定する
ax.plot(t,x,'-',c='k',label="x=sin(t)") #黒い実線
ax.plot(t,x1,'--',c='b',label="x=sin(t)+1") #青い実線
ax.plot(t,x2,'-.',c='r',label="x=sin(t)-1") #赤い点破線
ax.legend()

plt.show()