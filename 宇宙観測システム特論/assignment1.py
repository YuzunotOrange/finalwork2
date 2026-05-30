import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 定数
# -----------------------------
sigma = 5.67e-8
kpc_to_m = 3.086e19

# -----------------------------
# データ
# -----------------------------
data = {
    1: {"D": 4.5, "F": [2.13, 0.54, 0.42], "T": [2.15, 1.25, 1.25]},
    2: {"D": 10.9, "F": [0.74, 0.44, 0.28], "T": [2.32, 1.99, 1.77]},
    3: {"D": 4.2, "F": [5.20, 1.48, 0.42], "T": [2.70, 1.79, 1.26]},
    4: {"D": 7.2, "F": [0.39, 0.14], "T": [1.75, 1.17]},
    5: {"D": 7.4, "F": [1.10, 0.35], "T": [2.33, 1.88]},
    6: {"D": 8.0, "F": [1.08, 0.41], "T": [1.84, 1.10]},
    7: {"D": 6.0, "F": [3.45, 0.91, 0.69], "T": [2.73, 2.07, 1.91]},
    8: {"D": 7.0, "F": [0.52, 0.17], "T": [1.86, 1.21]},
    9: {"D": 7.6, "F": [0.57, 0.21], "T": [1.65, 1.26]},
    10: {"D": 8.4, "F": [0.51, 0.11], "T": [1.82, 1.42]},
}

number = []
mean_R = []
std_R = []

all_results = []

for num, d in data.items():
    
    D = d["D"] * kpc_to_m
    
    radii = []

    for i, (F, T) in enumerate(zip(d["F"], d["T"])):
        F = F * 1e-11
        T = T * 1e7

        L = 4 * np.pi * D ** 2 * F
        W = sigma * T **4
        R = np.sqrt(L/(4 * np.pi * W))

        R_km = R / 1000

        radii.append(R_km)
    
    #平均
    mean = np.mean(radii)
    #標準偏差
    std = np.std(radii, ddof=1)

    number.append(num)
    mean_R.append(mean)
    std_R.append(std)

    print(f"\n --DATA {num}-- ")
    print("Radius [km] =", np.round(radii, 2))
    print(f"Mean Radius = {mean:.2f} km")
    print(f"Standard Deviation = {std:.2f} km")

    all_results.append({
        "Data" : num,
        "Radius [km]" : np.round(radii, 2),
        "Mean [km]" : round(mean, 2),
        "Std [km]" : round(std, 2)
    })

df = pd.DataFrame(all_results)

# CSV保存
df.to_csv("neutron_star_radius_results.csv", index=False)


# χ^2 検定

mean_R = np.array(mean_R)
std_R = np.array(std_R)

R0 = np.mean(mean_R)

# χ^2
chi2 = np.sum((mean_R - R0)**2 / std_R**2)

nu = len(mean_R) - 1

# reduced χ^2
chi2_reduced = chi2 / nu

print("\n===== Chi-square Test =====")
print(f"Global Mean Radius = {R0:.2f} km")
print(f"Chi-square = {chi2:.2f}")
print(f"Degrees of Freedom = {nu}")
print(f"Reduced Chi-square = {chi2_reduced:.2f}")

plt.figure(figsize=(8,5))

plt.errorbar(
    number,
    mean_R,
    yerr=std_R,
    fmt='o',
    capsize=5
)

plt.xlabel("Source Number")
plt.ylabel("Radius [km]")

plt.title("Neutron Star Radius")

plt.xticks(number)
plt.grid()

# 画像保存
plt.savefig("neutron_star_radius_plot.png")

# 表示
plt.show()