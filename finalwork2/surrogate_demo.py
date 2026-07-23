import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import sys
import datetime
import random


#data = data[:200:1]


def rssurrogate(x):
    x_random = random.sample(x,len(x))

    #Calculating and plotting autocorrelation
    autocorrelation_plot(pd.Series(x))
    autocorrelation_plot(pd.Series(x_random))
    return (x)

def ftsurrogate(x):
    x = np.asarray(x)
    y = np.fft.fft(x)
    n = len(x)

    if n % 2 == 0:
        l = n // 2 - 1
        r = np.exp(2j * np.pi * np.random.rand(l))
        v = np.concatenate(([1], r, [1], np.flip(np.conj(r))))
    else:
        l = (n - 1) // 2
        r = np.exp(2j * np.pi * np.random.rand(l))
        v = np.concatenate(([1], r, np.flip(np.conj(r))))

    z = np.fft.ifft(y * v)
    return np.real(z)

# def histo(y,bin)

# #import numpy as np
# time = 100
# mean = 0.0
# std  = 1.0
# white_noise = np.random.normal(mean,std,time)
# white_noise = list(white_noise)

# random_white_noise = random.sample(white_noise, len(white_noise))

# autocorrelation_plot(pd.Series(random_white_noise))
# autocorrelation_plot(pd.Series(white_noise))

# plt.show()

# f_s = 1 # サンプリングレート f_s[Hz] (任意)
# t_fin = 2049 # 収録終了時刻 [s] (任意)
# dt = 1/f_s # サンプリング周期 dt[s]
# N = int(f_s * t_fin) # サンプル数 [個]

# file input
filename = sys.argv[1]
x= np.loadtxt(filename, dtype='float',usecols=[0],encoding='utf-8-sig')
print(len(x))

hindo=[]
for kk in range(0,39):
    y=ftsurrogate(x)
    y_sum =0.0
    for i in range(0,len(y)-1):
        y_sum = y_sum + (y[i]**2 * y[i+1]**2)
    y_sum = y_sum /(len(y)-1)
    hindo.append(y_sum)
# saidai=max(hindo)
# saishou=min(hindo)
# sorted_hindo=sorted(hindo)

x_sum =0.0
for i in range(0,len(x)-1):
    x_sum= x_sum + (x[i]**2 * x[i+1]**2)
x_sum = x_sum /(len(x)-1)

fig,ax=plt.subplots()
ymin=0
ymax=10

ax.hist(hindo)
ax.vlines([x_sum],ymin,ymax,"red", linestyles='dashed')
plt.show()


# y_fft = np.fft.fft(y) # 離散フーリエ変換
# y_freq = np.fft.fftfreq(N, d=dt) # 周波数を割り当てる（※後述）
# y_Amp = abs(y_fft/(N/2)) # 音の大きさ（振幅の大きさ）

# x_fft=np.fft.fft(data)
# x_freq = np.fft.fftfreq(N, d=dt)
# x_Amp= abs(x_fft/(N/2))

# fig,ax=plt.subplots(4)

# ax[0].plot(data)
# ax[1].plot(x_freq[1:int(N/2)], x_Amp[1:int(N/2)]) # A-f グラフのプロット
# ax[1].set_xscale("log") # 横軸を対数軸にセット

# ax[2].plot(y)
# ax[3].plot(y_freq[1:int(N/2)], y_Amp[1:int(N/2)]) # A-f グラフのプロット
# ax[3].set_xscale("log") # 横軸を対数軸にセット

plt.show()