import numpy as np
import matplotlib.pyplot as plt

# Define the step response function
def step_response(t):
    return (2/5) * (1 - 4 * np.exp(-5/4 * t))

# Generate time values
t = np.linspace(0, 5, 500)

# Calculate the step response
h_t = step_response(t)

# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(t, h_t, label='h(t) = (2/5) * (1 - 4 * exp(-5/4 * t))', color='b')
plt.axhline(y=2/5, color='r', linestyle='--', label='Final Value = 2/5')
plt.xlabel('Time (t)')
plt.ylabel('h(t)')
plt.title('Step Response of 1st Order System H(s) = 2 / (4s + 5)')
plt.legend()
plt.grid(True)
plt.show()
