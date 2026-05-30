import numpy as np
import matplotlib.pyplot as plt

def euler(t,x,f,h):
    """
    t:時間
    x:解
    f:関数dx/dt=f
    h:刻み幅
    """
    return x+h*f(t,x)

def func(t,x):
    """
    dx/dt=x
    """
    return x

#Initialize
N=20
X = np.zeros(N+1)
t=0.0
x=1
X[0]=x
h=0.05

#Calculate
for i in range(N):
    x=euler(t,x,func,h)
    X[i+1] = x
    t=(i+1)*h

#Graph
fig =plt.figure()
xx=np.linspace(0,1,256)
plt.plot(xx,np.exp(xx),'--')
plt.plot(np.linspace(0,1,N+1),X,'o-')

# --- 誤差の出力用に追加 ---
true_value = np.exp(1.0)          # t=1.0 における真の解 (e^1)
calc_value = X[N]                 # オイラー法で計算した最後の値
error = abs(true_value - calc_value) # 絶対誤差の計算

print(f"t=1.0 での真の解: {true_value:.6f}")
print(f"t=1.0 での近似解: {calc_value:.6f}")
print(f"絶対誤差: {error:.6f}")

plt.show()