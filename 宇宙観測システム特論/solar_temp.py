import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 6.626e-27
k = 1.38e-16
c = 3.0e10

def blackbody_lambda(lam_cm, T):
    return (2*h*c**2 / lam_cm**5) / (np.exp(h*c/(lam_cm*k*T)) - 1)


df = pd.read_excel("課題3.xlsx")
df.columns = df.columns.str.strip()

lambda_um = pd.to_numeric(df["波長"], errors="coerce")
log_I = pd.to_numeric(df["logI"], errors="coerce")

mask = (~lambda_um.isna()) & (~log_I.isna())

lambda_um = lambda_um[mask].to_numpy()
log_I = log_I[mask].to_numpy()


lambda_cm = lambda_um * 1e-4
I_cm = (10**log_I) * 1e4


lambda_max = lambda_cm[np.argmax(I_cm)]
Tw = 0.29 / lambda_max

print(f"Tw = {Tw:.1f} K")

lambda_plot = np.linspace(lambda_cm.min(), lambda_cm.max(), 1000)
B_Tw = blackbody_lambda(lambda_plot, Tw)


plt.figure(figsize=(8,6))

plt.scatter(lambda_cm, I_cm, label="Observed Data")
plt.plot(lambda_plot, B_Tw, label=f"Tw = {Tw:.0f} K")

# ピーク周辺だけ表示
plt.xlim(0, 50.0e-5)
plt.ylim(0, 4.5e14)
plt.xlabel(r"Wavelength $\lambda$ [cm]")
plt.ylabel(r"Intensity $I_{\lambda}$ [erg/(s cm$^2$ sterad cm)]")
plt.title("Observed Solar Spectrum and Color Temperature")
plt.grid(True)
plt.legend()

plt.show()