import numpy as np
import matplotlib.pyplot as plt

# Define the transfer function G(s) = 3 / (s + 2)
def G(s):
    return 3 / (s + 2)

# Define the frequency range
omega = np.logspace(-2, 2, 500)  # from 0.01 to 100 rad/s

# Calculate the magnitude in dB
magnitude = 20 * np.log10(np.abs(G(1j * omega)))

# Define the linear approximation
# For frequencies below 2 rad/s, the gain is approximately 7.6 dB
# For frequencies above 2 rad/s, the gain decreases at -20 dB/decade

# Linear approximation
approx_gain = []
for w in omega:
    if w < 2:
        approx_gain.append(7.6)
    else:
        approx_gain.append(7.6 - 20 * np.log10(w / 2))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.semilogx(omega, magnitude, label='Exact Bode Plot')
plt.semilogx(omega, approx_gain, '--', label='Linear Approximation')
plt.title('Bode Plot of the Transfer Function G(s) = 3 / (s + 2)')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()
