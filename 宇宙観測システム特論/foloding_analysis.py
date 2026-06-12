#フォールディング解析の実装
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
data = pd.read_csv(
    "pulsar_lightcurve.csv",
    header=None,
    names=["time", "count"]
)

time = data["time"].values
count = data["count"].values

dt = 0.125

# Folding

def fold_lightcurve(time, count, T, dt=0.125):

    Nbin = int(T / dt)

    phase = time % T

    indices = (phase / dt).astype(int)

    indices = np.clip(indices, 0, Nbin - 1)

    folded_sum = np.bincount(
        indices,
        weights=count,
        minlength=Nbin
    )

    folded_num = np.bincount(
        indices,
        minlength=Nbin
    )

    xi = np.zeros(Nbin)

    valid = folded_num > 0

    xi[valid] = (
        folded_sum[valid]
        / folded_num[valid]
    )

    sigma = np.zeros(Nbin)

    sigma[valid] = (
        np.sqrt(
            xi[valid] * folded_num[valid]
        )
        / folded_num[valid]
    )

    return xi, sigma


# reduced x^2

def reduced_chi2(xi, sigma):

    valid = sigma > 0

    x = xi[valid]
    s = sigma[valid]

    mean_x = np.mean(x)

    chi2 = np.sum(
        ((x - mean_x) ** 2)
        / (s ** 2)
    )

    return chi2 / (len(x) - 1)


#周期探索

trial_periods = np.arange(
    3.000,
    7.000 + dt,
    dt
)

chi2_values = []

best_period = None
best_chi2 = -np.inf

best_xi = None
best_sigma = None

for T in trial_periods:

    xi, sigma = fold_lightcurve(
        time,
        count,
        T,
        dt
    )

    chi2_red = reduced_chi2(
        xi,
        sigma
    )

    chi2_values.append(
        chi2_red
    )

    if chi2_red > best_chi2:

        best_chi2 = chi2_red
        best_period = T

        best_xi = xi.copy()
        best_sigma = sigma.copy()



print(f"Best Period = {best_period:.3f} s")
print(f"Maximum reduced χ² = {best_chi2:.2f}")



# Step2
# reduced χ² vs Trial Period

plt.figure(figsize=(8,5))

plt.step(
    trial_periods,
    chi2_values,
    where='mid',
    linewidth=1.5
)

plt.axvline(
    best_period,
    linestyle='--',
    label=f'Best Period = {best_period:.3f} s'
)

plt.xlabel('Trial Period [s]')
plt.ylabel(r"Reduced $\chi^2$")

plt.title('Epoch Folding Analysis')

plt.legend()

plt.tight_layout()
plt.show()


phase_time = np.arange(
    len(best_xi)
) * dt

plt.figure(figsize=(8,5))

plt.step(
    phase_time,
    best_xi,
    where='mid',
    linewidth=1.5
)

plt.xlabel('Phase Time [s]')
plt.ylabel('Average Counts')

plt.title(
    f'Folded Light Curve (T = {best_period:.3f} s)'
)

plt.tight_layout()
plt.show()