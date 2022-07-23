import matplotlib.pyplot as plt
import numpy as np
import sys, os, argparse

bins = 200
buffer = 0.1

filenames = sys.argv[1:]
L = 0

reflection_set = set()
timestep_set = set()

for filename in filenames:
    #Get metadata
    vars = filename.split('_')
    reflection = vars[1]
    reflection_set.add(reflection)
    timestep = vars[2]
    timestep_set.add(timestep)
    particles = vars[3]
    time = vars[4]
    L = float(vars[5])

    #Create label from metadata
    lab = reflection + ", dt=" + timestep + ", Np=" + particles

    #Load data
    Z = np.load(filename)

    #Put data into histogram
    hist = np.histogram(Z, bins, (0, L))
    depth = hist[1][:-1]
    C = hist[0] / (np.sum(hist[0]) * (depth[1] - depth[0]))

    plt.plot(C, depth, label=lab)


out_filename = ""
for ref in reflection_set:
    out_filename += ref
out_filename += "_"
for step in timestep_set:
    out_filename += step
out_filename += ".png"

plt.ylim(L + buffer, -buffer)
plt.xlim(0.4, 0.6)
plt.xlabel("Concentration")
plt.grid()
plt.legend()

plt.savefig(out_filename)
