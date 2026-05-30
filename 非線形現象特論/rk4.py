import numpy as np
import matplotlib.pyplot as plt

def rk4(t,x,f,h):
    """
    t:時間
    x:解
    f:関数dx/dt=f
    h:刻み幅
    """
    k1=f(t,x)
    xtemp=x+h/2.0*k1
    k2=f(t,xtemp)
    xtemp=x+h/2.0*k2
    k3=f(t,xtemp)
    xtemp=x+h*k3
    k4=f(t,xtemp)
    return x+(k1+2.0*(k2+k3)+k4)*h/6.0

def func(t,x):
    """
    dx/dt=x
    """
    return x

#Initialize
N=10
X = np.zeros(N+1)
t=0.0
x=1
X[0]=x
h=0.1

#Calculate
for i in range(N):
    x=rk4(t,x,func,h)
    X[i+1] = x
    t=(i+1)*h

print(X)    
#Graph
fig =plt.figure()
xx=np.linspace(0,1,256)
plt.plot(xx,np.exp(xx),'--')
plt.plot(np.linspace(0,1,N+1),X,'o-')

plt.show()