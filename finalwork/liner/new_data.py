import numpy as np
import matplotlib.pyplot as plt

# ファイル読み込み
#scan data
new_data = pd.read_csv(args.csv_files, header=None, delim_whitespace=True)
series = new_data.iloc[:, 0].values

# 列を分離
time = new_data[:, 0]
flux = new_data[:, 1]
error = new_data[:, 2]

# 時間を週に変換（任意）
time_week = (time - time[0]) / 7

# 点でプロット（補間なし）
plt.figure()
plt.plot(time_week, flux, 'o', color='blue', label='Observed Data')
plt.xlabel("Time [weeks]")
plt.ylabel("Flux")
plt.title("Original Light Curve (No Interpolation)")
plt.grid(True)
plt.legend()
plt.show()
