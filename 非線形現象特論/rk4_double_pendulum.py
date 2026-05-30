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
    k2=f(t + h/2.0, x + h / 2.0 * k1)
    k3=f(t + h/2.0, x + h / 2.0 * k2)
    k4=f(t + h, x + h * k3)
    return x+(k1+2.0*(k2+k3)+k4)*h/6.0

# --二重振り子の運動方程式--
g = 9.8
L = 0.3
m1 = 1.0
m2 = 1.0

def func(t, x):

    theta1 = x[0]
    omega1 = x[1]
    theta2 = x[2]
    omega2 = x[3]

    delta = theta1 - theta2

    #dθ/dt
    dtheta1 = omega1
    dtheta2 = omega2

    #dω/dt
    denom = 2 - np.cos(2 * delta)

    domega1 = ( -g * (2 * np.sin(theta1) + np.sin(theta1 -2 * theta2))
               -2 * np.sin(delta) * (omega2**2 * L + omega1**2 * L * np.cos(delta))) / (L * denom)
    
    domega2 = ( 2 * np.sin(delta) * (2 * omega1**2 * L + 2 * g * np.cos(theta1) + omega2**2 * L * np.cos(delta)))  / (L * denom)

    return np.array([dtheta1, domega1, dtheta2, domega2])

# --初期条件--
theta1_0 = np.pi
omega1_0 = 0.0

theta2_0 = -np.pi / 4
omega2_0 = 0.0

x = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

h = 0.06
T = 20.0
N = int (T / h)

x2_list = []
y2_list = []

t = 0.0

# --時間発展--
for i in range(N):
    x = rk4(t, x, func, h)
    t += h
    
    theta1 = x[0]
    theta2 = x[2]

    x1 = L * np.sin(theta1)
    y1 = -L * np.cos(theta1)

    x2 = x1 + L * np.sin(theta2)
    y2 = y1 - L * np.cos(theta2)

    x2_list.append(x2)
    y2_list.append(y2)

plt.figure(figsize=(6, 6))

plt.plot(x2_list, y2_list)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory of Mass 2 Double Pendulum")
plt.axis("equal")
plt.grid()
plt.show() 
