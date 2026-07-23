import sys
import os
import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#処理対象のファルダ
target_folder = "./liner"
csv_files = glob.glob(os.path.join(target_folder, "w_comae_lc.csv"))

for fname in csv_files:
    print(f"Processing:{fname}")
    
    outname = os.path.splitext(os.path.basename(fname))[0] + '_liner.csv'
    outpath = os.path.join(target_folder, outname)
    y_data = np.loadtxt(fname,usecols=[1], delimiter=',')
    x_data = np.loadtxt(fname,usecols=[0], delimiter=',')
    x_data = (x_data-x_data[0]) / 7
    x_latent = np.arange(0,x_data[-1], 1)
    f_curve = interpolate.interp1d(x_data, y_data)
    f_curve = interpolate.interp1d(x_data, y_data, kind="nearest")
    f_curve = interpolate.PchipInterpolator(x_data, y_data)
    print(x_latent)
    
    np.savetxt(outpath, f_curve(x_latent))
    fig = plt.figure()
    plt.plot(x_latent,f_curve(x_latent),"-o", color="red")
    plt.plot(x_data,y_data, "o", color="blue")
    plt.xlabel('Time[weeks]')
    plt.ylabel('Flux')
    plt.grid(True)
    plt.show()