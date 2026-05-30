import sys
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

fname = sys.argv[1]

outname = os.path.splitext(os.path.basename(fname))[0] + '_liner.csv'
print(outname)
y_data = np.loadtxt(fname,usecols=[1], delimiter=',')
x_data = np.loadtxt(fname,usecols=[0], delimiter=',')
x_data = (x_data-x_data[0]) / 7
x_latent = np.arange(0,x_data[-1], 1)
f_curve = interpolate.interp1d(x_data, y_data)
f_curve = interpolate.interp1d(x_data, y_data, kind="nearest")
f_curve = interpolate.PchipInterpolator(x_data, y_data)

print(x_latent)

np.savetxt(outname, f_curve(x_latent))

fig = plt.figure()
plt.plot(x_latent,f_curve(x_latent),"-o", color="red")
plt.plot(x_data,y_data, "o", color="blue")
plt.grid()
plt.show()