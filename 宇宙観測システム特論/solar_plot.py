#宇宙観測システム特論Ⅱ
#BP22104
#松本優瑞
#太陽の実測データ（離散プロット）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "課題3.xlsx"

df = pd.read_excel(file_name)

# 列名の余分な空白を削除
df.columns = df.columns.str.strip()

print(df.head())
print(df.dtypes)

da_um = pd.to_numeric(df["波長"], errors="coerce")
log_I = pd.to_numeric(df["logI"], errors="coerce")

mask = (~lambda_um.isna()) & (~log_I.isna())

lambda_um = lambda_um[mask].to_numpy()
log_I = log_I[mask].to_numpy()

I_um = 10**log_I

lambda_cm = lambda_um * 1e-4

I_cm = I_um * 1e4

plt.figure(figsize=(8,6))

plt.scatter(
    lambda_cm,
    I_cm,
    color="blue",
    s=35,
    label="Observed Data"
)

plt.xlabel(r"Wavelength $\lambda$ [cm]")
plt.ylabel(r"Intensity $I_{\lambda}$ [erg/(s cm$^2$ sterad cm)]")
plt.title("Observed Solar Spectrum")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()