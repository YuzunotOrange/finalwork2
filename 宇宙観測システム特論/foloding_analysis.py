import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# データ読み込み
# ======================

data = pd.read_csv(
    "pulsar_lightcurve.csv",
    header=None,
    names=["time", "count"]
)

time = data["time"].values
count = data["count"].values

# ======================
# パラメータ
# ======================

dt = 0.125

T_values = np.arange(3.0, 7.0 + dt, dt)

chi2_list = []

best_folded = None
best_T = None
best_chi2 = -1

# ======================
# trial period loop
# ======================

for T in T_values:

    # 位相
    phase = time % T

    # 位相ビン数
    Nbin = int(T / dt)

    # ビン番号
    indices = (phase / dt).astype(int)

    folded = np.zeros(Nbin)
    counts_per_bin = np.zeros(Nbin)

    # folding
    for i in range(len(count)):
        b = indices[i]

        folded[b] += count[i]
        counts_per_bin[b] += 1

    # 平均
    valid = counts_per_bin > 0
    folded[valid] /= counts_per_bin[valid]

    # 誤差
    sigma = np.sqrt(folded / counts_per_bin)

    # 全体平均
    mean_value = np.mean(folded[valid])

    # reduced chi2
    chi2 = np.sum(
        ((folded[valid] - mean_value) ** 2) /
        (sigma[valid] ** 2)
    ) / (Nbin - 1)

    chi2_list.append(chi2)

    # 最大値保存
    if chi2 > best_chi2:
        best_chi2 = chi2
        best_T = T
        best_folded = folded.copy()

# ======================
# 結果表示
# ======================

print("Best Period =", best_T)
print("Max reduced chi2 =", best_chi2)

# ======================
# chi2 vs period
# ======================

plt.figure(figsize=(8,5))
plt.plot(T_values, chi2_list)
plt.xlabel("Trial Period [s]")
plt.ylabel("Reduced Chi-square")
plt.title("Folding Analysis")
plt.grid()
plt.show()

# ======================
# folded light curve
# ======================

phase_axis = np.arange(len(best_folded)) * dt

plt.figure(figsize=(8,5))
plt.plot(phase_axis, best_folded)
plt.xlabel("Phase Time [s]")
plt.ylabel("Counts")
plt.title(f"Folded Light Curve (T={best_T:.3f} s)")
plt.grid()
plt.show()