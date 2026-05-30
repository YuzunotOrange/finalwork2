import numpy as np
import matplotlib.pyplot as plt

# Given parameters
L = 2.2e-3  # inductance in Henrys
C = 0.47e-6  # capacitance in Farads
R = 30  # resistance in Ohms

# Frequency range
f = np.logspace(3, 5, num=500)  # from 1 kHz to 100 kHz
omega = 2 * np.pi * f  # angular frequency

# Calculate impedance
Z = np.sqrt(R**2 + (omega * L - 1 / (omega * C))**2)

# Plot impedance on a logarithmic scale
plt.figure(figsize=(10, 6))
plt.loglog(f, Z)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (|Z|) (Ohms)')
plt.title('Impedance of Series Resonant Circuit')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(10, 1e3)
plt.show()
