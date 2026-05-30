import numpy as np
import matplotlib.pyplot as plt

# Define the transfer function G(s) = (s + 10) / (10s + 1)
def G(s):
    return (s + 10) / (10 * s + 1)

# Define the frequency range
omega = np.logspace(-2, 2, 500)  # from 0.01 to 100 rad/s

# Calculate the magnitude in dB
magnitude = 20 * np.log10(np.abs(G(1j * omega)))

# Calculate the phase in degrees
phase = np.angle(G(1j * omega), deg=True)

# Define the linear approximation for magnitude
approx_magnitude = []
for w in omega:
    if w < 0.1:
        approx_magnitude.append(20)
    elif w < 10:
        approx_magnitude.append(20 - 20 * np.log10(w / 0.1))
    else:
        approx_magnitude.append(0)

# Plot the Bode magnitude plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogx(omega, magnitude, label='Exact Bode Plot')
plt.semilogx(omega, approx_magnitude, '--', label='Linear Approximation')
plt.title('Bode Plot of the Transfer Function G(s) = (s + 10) / (10s + 1)')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Plot the Bode phase plot
plt.subplot(2, 1, 2)
plt.semilogx(omega, phase, label='Exact Bode Plot')
plt.axvline(x=0.1, color='r', linestyle='--', linewidth=0.7, label='Pole at 0.1 rad/s')
plt.axvline(x=10, color='g', linestyle='--', linewidth=0.7, label='Zero at 10 rad/s')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
