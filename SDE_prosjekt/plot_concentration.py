import matplotlib.pyplot as plt
import numpy as np
import sys, os, argparse

bins = 200
buffer = 0.1

filenames = sys.argv[1:]
L = 0

for filename in filenames:
    #Get metadata
    vars = filename.split('_')
    reflection = vars[1]
    timestep = vars[2]
    particles = vars[3]
    time = vars[4]
    L = float(vars[5])

    #create label from metadata
    lab = reflection + ", dt=" + timestep + ", Np=" + particles

    #Load data
    Z = np.load(filename)

    #Put data into histogram
    hist = np.histogram(Z, bins, (0, L))
    depth = hist[1][:-1]
    C = hist[0] / (np.sum(hist[0]) * (depth[1] - depth[0]))

    plt.plot(C, depth, label=lab)


plt.ylim(L + buffer, -buffer)
plt.xlim(0.4, 0.6)
plt.xlabel("Concentration")
plt.grid()
plt.legend()

plt.savefig("plot.png")
