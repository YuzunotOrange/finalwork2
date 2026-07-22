#宇宙観測システム特論Ⅱ
#BP22104
#松本優瑞
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# cgs単位
h = 6.626e-27
k = 1.38e-16
c = 3.0e10
sigma = 5.67e-5

def blackbody_lambda(lam_cm, T):
    x = h * c / (lam_cm * k * T)
    return (2 * h * c**2 / lam_cm**5) / (np.exp(x) - 1)


file_name = "課題3.xlsx"

df = pd.read_excel(file_name)
df.columns = df.columns.str.strip()

# A列・B列のみ
df["波長"] = pd.to_numeric(df["波長"], errors="coerce")
df["logI"] = pd.to_numeric(df["logI"], errors="coerce")

df = df.dropna(subset=["波長", "logI"])

lambda_um = df["波長"].to_numpy(dtype=float)
log_I = df["logI"].to_numpy(dtype=float)

I_um = 10 ** log_I
lambda_cm = lambda_um * 1e-4
I_obs = I_um * 1e4

sort_idx = np.argsort(lambda_cm)
lambda_cm = lambda_cm[sort_idx]
I_obs = I_obs[sort_idx]


max_index = np.argmax(I_obs)
lambda_max = lambda_cm[max_index]
Tw = 0.29 / lambda_max


# λ = 1.0e-4 cm に近い点を使用
target_lambda = 1.0e-4
idx = np.argmin(np.abs(lambda_cm - target_lambda))

lambda_tb = lambda_cm[idx]
I_tb = I_obs[idx]

Tb = (h * c) / (
    lambda_tb * k *
    np.log((2 * h * c**2) / (lambda_tb**5 * I_tb) + 1)
)

I_total = np.trapz(I_obs, lambda_cm)
Teff = ((np.pi * I_total) / sigma) ** 0.25

print("結果")
print(f"lambda_max = {lambda_max:.3e} cm")
print(f"Tw = {Tw:.1f} K")
print()
print(f"Tbに使った波長 = {lambda_tb:.3e} cm")
print(f"Tbに使ったIλ = {I_tb:.3e}")
print(f"Tb = {Tb:.1f} K")
print()
print(f"I_total = {I_total:.3e} erg/(s cm^2 sterad)")
print(f"Teff = {Teff:.1f} K")

os.makedirs("output", exist_ok=True)

lambda_plot = np.linspace(lambda_cm.min(), lambda_cm.max(), 2000)

B_Tw = blackbody_lambda(lambda_plot, Tw)
B_Tb = blackbody_lambda(lambda_plot, Tb)
B_Teff = blackbody_lambda(lambda_plot, Teff)

plt.figure(figsize=(9, 6))

plt.scatter(lambda_cm, I_obs, label="Observed solar data", s=35)
plt.plot(lambda_plot, B_Tw, label=f"Blackbody Tw = {Tw:.0f} K")
plt.plot(lambda_plot, B_Tb, label=f"Blackbody Tb = {Tb:.0f} K")
plt.plot(lambda_plot, B_Teff, label=f"Blackbody Teff = {Teff:.0f} K")

plt.xlabel(r"Wavelength $\lambda$ [cm]")
plt.ylabel(r"Intensity $I_\lambda$ [erg/(s cm$^2$ sterad cm)]")
plt.title("Solar spectrum and blackbody radiation")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("output/final_graph.png", dpi=300)
plt.savefig("output/final_graph.pdf")
plt.show()