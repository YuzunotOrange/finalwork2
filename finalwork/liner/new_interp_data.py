import sys
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python script.py <filename.csv>")
    sys.exit(1)

fname = sys.argv[1]
if not os.path.isfile(fname):
    print(f"Error: File '{fname}' not found.")
    sys.exit(1)

print(f"Processing: {fname}")

x_data = np.loadtxt(fname, usecols=[0], delimiter=',')
y_data = np.loadtxt(fname, usecols=[1], delimiter=',')

x_data = (x_data - x_data[0]) / 7
x_latent = np.arange(0, x_data[-1], 1)

f_curve = interpolate.PchipInterpolator(x_data, y_data)

base = os.path.splitext(os.path.basename(fname))[0]
outname = base + '_liner.csv'
outpath = os.path.join(os.path.dirname(fname), outname)

np.savetxt(outpath, f_curve(x_latent), delimiter=',')

plt.figure()
plt.plot(x_latent, f_curve(x_latent), "-o", color="red", label="Interpolated")
plt.plot(x_data, y_data, "o", color="blue", label="Original")
plt.xlabel("Time (normalized)")
plt.ylabel("Value")
plt.title(outname)
plt.grid()
plt.legend()
plt.show()
