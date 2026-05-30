#Hodgkin-Huxley 方程式
#課題1
import numpy as np
import matplotlib.pyplot as plt

class Hodgkinhuxley:
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
        m = 0.1
        h = 0.1
        n = 0.1

        v_monitor = []
        m_monitor = []
        h_monitor = []
        n_monitor = []

        time = int(self.time / self.dt)

        #update time
        for t in range(time):
            #calc channel gating kinetics
            m += self.dm(v, m)
            h += self.dh(v, h)
            n += self.dn(v, n)
            dv = (
                self.gNa * m**3 * h * (self.ENa - v)  
                + self.gK * n**4 * (self.EK - v) 
                + (self.gl * (self.EL - v)) 
                + (i[t] / self.Cm)
            )
            #clac new membrane potetial
            v += dv * self.dt
            #record
            v_monitor.append(v)
            m_monitor.append(m)
            h_monitor.append(h)
            n_monitor.append(n)
        return v_monitor, m_monitor, h_monitor, n_monitor,
    
    
    def dm(self, v, m):
        return (self.alpha_m(v) * (1 - m) - self.beta_m(v) * m) * self.dt
    
    def dh(self, v, h):
        return (self.alpha_h(v) * (1 - h) - self.beta_h(v) * h) * self.dt
    
    def dn(self, v, n):
        return (self.alpha_n(v) * (1 - n) - self.beta_n(v) * n) * self.dt
    
    def alpha_m(self, v):
        return (0.1 * (v + 40)) / (1 - np.exp(-(v + 40) /10))
    
    def alpha_h(self, v):
        return 0.07 * np.exp(-(v + 65) / 20)
    
    def alpha_n(self, v):
        return (0.01 * (v + 55)) / (1 - np.exp(-(v + 55) / 10))
    
    def beta_m(self, v):
        return 4 * np.exp(- (v + 65) / 18)
    
    def beta_h(self, v):
        return 1 / (1 + np.exp(-(v + 35) / 10))
    
    def beta_n(self, v):
        return 0.125 * np.exp(-(v + 65) / 80)
    
if __name__ == '__main__':
    #init experimental time and time-step
    time = (1/64)*16384
    dt = 1/64
    #neuあとで調べること
    neu = Hodgkinhuxley(time, dt) 

    #the data
    input_data_3 = np.full(int(time / dt), 3)
    input_data_10 = np.full(int(time / dt), 10)

    #calculate the neuron each situation
    v_3, m_3, h_3, n_3, = neu.calc_v(input_data_3)
    v_10, m_10, h_10, n_10,= neu.calc_v(input_data_10)

    #plot
    x = np.arange(0, time, dt)
    plt.figure(figsize=(12, 8))

    #I = 3 
    plt.subplot(4, 2, 1)
    plt.plot(x, v_3)
    plt.title('I = 3 μA/cm^2: Membrane Potential')
    plt.ylabel('V [mV]')
    plt.ylim(-75,25)
    plt.yticks(np.arange(-75, 30, 25))

    plt.subplot(4, 2, 3)
    plt.plot(x, m_3, label='m')
    plt.title('I = 3 μA/cm^2: State variables')
    plt.ylabel('m')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))


    plt.subplot(4, 2, 5)
    plt.plot(x, h_3, label='h')
    plt.title('I = 3 μA/cm^2: State variables')
    plt.ylabel('h')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))


    plt.subplot(4, 2, 7)
    plt.plot(x, n_3, label='n')
    plt.title('I = 3 μA/cm^2: State variables')
    plt.ylabel('n')
    plt.xlabel('Time [ms]')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))


    plt.legend()



    
    #I = 10
    plt.subplot(4, 2, 2)
    plt.plot(x, v_10)
    plt.title('I = 10 μA/cm^2: Membrane Potential')
    plt.ylabel('V [mV]')
    plt.ylim(-75,25)
    plt.yticks(np.arange(-75, 30, 25))
    
    plt.subplot(4, 2, 4)
    plt.plot(x, m_10, label='m')
    plt.title('I = 10 μA/cm^2: State variables')
    plt.ylabel('m')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))


    plt.subplot(4, 2, 6)
    plt.plot(x, h_10, label='h')
    plt.title('I = 10 μA/cm^2: State variables')
    plt.ylabel('h')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))

    plt.subplot(4, 2, 8)
    plt.plot(x, n_10, label='n')
    plt.title('I = 10 μA/cm^2: State variables')
    plt.ylabel('n')
    plt.xlabel('Time [ms]')
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.01, 0.25))


    plt.tight_layout()
    plt.show()




        


        