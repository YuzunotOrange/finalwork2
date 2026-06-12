import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data


data = pd.read_csv(
    "pulsar_lightcurve.csv",
    header=None,
    names=["time","count"]
)

count = data["count"].values

dt = 0.125


# FFT


fft_result = np.fft.rfft(count)

power = np.abs(fft_result)**2

freq = np.fft.rfftfreq(
    len(count),
    d=dt
)

# DC成分除去
freq = freq[1:]
power = power[1:]

# 最大ピーク
peak_idx = np.argmax(power)

peak_freq = freq[peak_idx]

peak_period = 1 / peak_freq

print(f"Peak Frequency = {peak_freq:.4f} Hz")
print(f"Estimated Period = {peak_period:.3f} s")


plt.figure(figsize=(8,5))

plt.plot(freq,power)

plt.axvline(
    peak_freq,
    ls='--',
    color='red',
    label=f'{peak_freq:.3f} Hz'
)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")

plt.title("Power Spectrum")

plt.legend()

plt.tight_layout()
plt.show()