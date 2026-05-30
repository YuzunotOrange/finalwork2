#Hodgkin-Huxley 方程式
#課題2
import numpy as np
import matplotlib.pyplot as plt

class Hodkinhuxley:
    def __init__(self, time, dt, rest=-60, Cm=1.0, gNa=120., gK=36., gL=0.3, ENa=50, EK=-77,EL=-54.4):
        self.time = time
        self.dt = dt
        self.rest = rest
        self.Cm = Cm
        self.gNa = gNa
        self.gK = gK
        self.gl =gL
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
    def calc_v(self, i):
        """ compute membrane potential """

        #initialize parameters
        v = self.rest
        n = 0.1
        m = 0.1
        h = 0.01

        v_monitor = []
        n_monitor = []
        m_monitor = []
        h_monitor = []

        time = int(self.time / self.dt)

        #update time
        for t in range(time):
            #calc channel gating kinetics
            n += self.dn(v, n)
            m += self.dm(v, m)
            h += self.dh(v, h)
            dv = (
                self.gNa * m**3 * h * (self.ENa - v)
             + self.gK * n**4 * (self.EK - v) 
             + self.gl * (self.EL - v) 
             + (i[t] / self.Cm)
            )
            #clac new membrane potetial
            v += dv * self.dt
            #record
            v_monitor.append(v)
            n_monitor.append(n)
            m_monitor.append(m)
            h_monitor.append(h)

        return v_monitor, n_monitor, m_monitor, h_monitor
    
    def dn(self, v, n):
        return (self.alpha_n(v) * (1 - n) - self.beta_n(v) * n) * self.dt
    
    def dm(self, v, m):
        return (self.alpha_m(v) * (1 - m) - self.beta_m(v) * m) * self.dt
    
    def dh(self, v, h):
        return (self.alpha_h(v) * (1 - h) - self.beta_h(v) * h) * self.dt
    
    def alpha_n(self, v):
        return (0.01 * (v + 55)) / (1 - np.exp(-(v + 55) / 10))
    
    def alpha_m(self, v):
        return (0.1 * (v + 40)) / (1 - np.exp(-(v + 40) /10))
    
    def alpha_h(self, v):
        return 0.07 * np.exp(-(v + 65) / 20)
    
    def beta_n(self, v):
        return 0.125 * np.exp(-(v + 65) / 80)
    
    def beta_m(self, v):
        return 4 * np.exp(- (v + 65) / 18)
    
    def beta_h(self, v):
        return 1 / (1 + np.exp(-(v + 35) / 10))
# 実行部
if __name__ == '__main__':
    time = (1/64)*16384
    dt = 1/64
    neu = Hodkinhuxley(time, dt)

    I_test = 10  # 発火する電流
    input_data = np.full(int(time / dt), I_test)

    v, m, h, n = neu.calc_v(input_data)

    # プロット
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(v)
    plt.title(f'Membrane Potential v(t) for I={I_test} μA/cm²')
    plt.ylabel('Membrane Potential (mV)')

    plt.subplot(2, 1, 2)
    plt.plot(h, color='orange')
    plt.title('Inactivation Variable h(t) of Na⁺ Channel')
    plt.xlabel('Time step')
    plt.ylabel('h (Inactivation)')

    plt.tight_layout()
    plt.show()
        