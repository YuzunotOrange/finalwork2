#宇宙観測システム特論Ⅱ
#課題
#BP22104　松本優瑞

import numpy as np
import matplotlib.pyplot as plt

#cgs単位
h = 6.626e-27 
k = 1.38e-16
c = 3.0e10

temps = [3, 300, 6000, 1e7]

def Bnu(nu, T):
    x = h * nu / (k * T)
    return (2 * h * nu**3 / c**2) / np.expm1(x)

def rayleigh_jeans(nu, T):
    return 2 * k * T * nu**2 / c**2

def wien(nu, T):
    x = h * nu / (k * T)
    return (2 * h * nu**3 / c**2) * np.exp(-x)

#真数
nu = np.logspace(6, 23, 3000)
#対数
nu_linear = np.linspace(1e5, 1e19, 10000)


#課題(1)
#黒体放射　真数表示
plt.figure(figsize=(8, 5))
for T in temps:
    plt.plot(nu_linear, Bnu(nu_linear, T), label=f"T = {T:g} K")

plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"$B_\nu$ [erg/(s cm$^2$ Hz sterad)]")
plt.title("Blackbody Radiation")
plt.legend()
plt.grid(True)
plt.xlim(0, 1e19)
plt.ylim(0, 2e5)
plt.savefig("blackbody_linear.png", dpi=300, bbox_inches="tight")
plt.show()

#課題(1)
#黒体放射　対数表示
plt.figure(figsize=(8, 5))
for T in temps:
    plt.loglog(nu, Bnu(nu, T), label=f"T = {T:g} K")

plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"$B_\nu$ [erg/(s cm$^2$ Hz sterad)]")
plt.title("Blackbody Radiation log-log")
plt.legend()
plt.grid(True, which="both")
plt.ylim(1e-40, 1e10)
plt.savefig("blackbody_loglog.png", dpi=300, bbox_inches="tight")
plt.show()

#課題(2)
#レイリージーンズ分布とウィーン分布真数表示
colors = {
    3: "blue",
    300: "orange",
    6000: "green",
    1e7: "red"
}

plt.figure(figsize=(8, 5))

for T in temps:
    color = colors[T]

    plt.plot(
        nu_linear, Bnu(nu_linear, T),
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"Planck T={T:g}K"
    )

    plt.plot(
        nu_linear, rayleigh_jeans(nu_linear, T),
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"RJ T={T:g}K"
    )

    plt.plot(
        nu_linear, wien(nu_linear, T),
        color=color,
        linestyle=":",
        linewidth=2,
        label=f"Wien T={T:g}K"
    )

plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"$B_\nu$ [erg/(s cm$^2$ Hz sterad)]")
plt.title("Blackbody Radiation with Approximations")
plt.legend(fontsize=7, ncol=2)
plt.grid(True)

plt.xlim(0, 1e19)
plt.ylim(0, 2e5)

plt.savefig("blackbody_approximations_linear.png", dpi=300, bbox_inches="tight")
plt.show()

#レイリージーンズ分布とウィーン分布対数表示
plt.figure(figsize=(8,5))
for T in temps:
    color = colors[T]

    # Planck
    plt.loglog(
        nu, Bnu(nu,T),
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"Planck T={T:g}K"
    )

    # Rayleigh-Jeans
    plt.loglog(
        nu, rayleigh_jeans(nu,T),
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"RJ T={T:g}K"
    )

    # Wien
    plt.loglog(
        nu, wien(nu,T),
        color=color,
        linestyle=":",
        linewidth=2,
        label=f"Wien T={T:g}K"
    )

plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"$B_\nu$ [erg/(s cm$^2$ Hz sterad)]")
plt.title("Blackbody Radiation with Approximations log-log")
plt.legend(fontsize=8, ncol=2)
plt.grid(True, which="both")
plt.ylim(1e-50, 1e40)
plt.savefig("blackbody_approximations.png", dpi=300, bbox_inches="tight")
plt.show()