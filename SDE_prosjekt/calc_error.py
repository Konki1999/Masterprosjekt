import numpy as np
import os
import matplotlib.pyplot as plt
from random_walk import random_walk, K_func, K_func_prime

dts = np.array([2,3,4,6,8,12,16])
Np = int(1e4)
T = 3600 * 6
L = 2

mean_list = np.zeros((2,len(dts)))

for i, dt in enumerate(dts):
    filename_s = f"results_s_{dt}_{Np}_{T/3600}_{L}_.npy"
    if not os.path.exists(filename_s):
        res_s = random_walk(Np, T, dt, K_func, K_func_prime, L, True)
        np.save(filename_s, res_s)
    else:
        res_s = np.load(filename_s)
    print(filename_s)

    mean_list[0,i] = np.mean(res_s, axis=1)

    filename_l = f"results_l_{dt}_{Np}_{T/3600}_{L}_.npy"
    if not os.path.exists(filename_l):
        res_l = random_walk(Np, T, dt, K_func, K_func_prime, L, False)
        np.save(filename_l, res_l)
    else:
        res_l = np.load(filename_l)
    print(filename_l)

    mean_list[1,i] = np.mean(res_l)

error_list = np.abs(mean_list - L/2)

A = np.vstack([np.log10(dts), np.ones(len(dts))]).T
ms, cs = np.linalg.lstsq(A, np.log10(error_list[0]), rcond=None)[0]
ml, cl = np.linalg.lstsq(A, np.log10(error_list[1]), rcond=None)[0]

print(f"{ms = }")
print(f"{ml = }")

plt.scatter(dts, error_list[0], c='r')
plt.plot(dts, np.power(10, cs) * np.power(dts, ms), 'r')

plt.scatter(dts, error_list[1], c='b')
plt.plot(dts, np.power(10, cl) * np.power(dts, ml), 'b')

plt.grid(color="grey", linestyle='--')

plt.xticks(dts)

plt.xscale('log')
plt.yscale('log')

plt.savefig("error.png")
