import numpy as np
import os
import matplotlib.pyplot as plt
from random_walk import random_walk, K_func, K_func_prime

dts = np.array([2,3,4,6,8,12,16])
Np = int(1e5)
T = 3600 * 6
L = 2
t_eq = 3600 * 0.5
save_interval = 100

r_eq = (T - t_eq) / T # Factor with which to reduce the number of timesteps to include in the error analysis
num_saves = int(np.floor(T / ( save_interval))) # Number of timesteps that are saved
saves_for_error_calc = int(np.floor(r_eq * num_saves)) # Include saves_for_error_calc last save points in calculation of error
mean_list = np.zeros((2,len(dts)))

for i, dt in enumerate(dts):
    filename_s = f"results_s_{dt}_{Np}_{T/3600}_{L}_.npy"
    if not os.path.exists(filename_s):
        res_s = random_walk(Np, T, dt, K_func, K_func_prime, L, True, save_interval)
        np.save(filename_s, res_s)
    else:
        res_s = np.load(filename_s)
    print(filename_s)

    # Take mean of position along all axis for the last saves_for_error_calc save points along time axis
    mean_list[0,i] = np.mean(res_s[-saves_for_error_calc:])

    filename_l = f"results_l_{dt}_{Np}_{T/3600}_{L}_.npy"
    if not os.path.exists(filename_l):
        res_l = random_walk(Np, T, dt, K_func, K_func_prime, L, False, save_interval)
        np.save(filename_l, res_l)
    else:
        res_l = np.load(filename_l)
    print(filename_l)

    mean_list[1,i] = np.mean(res_l[-saves_for_error_calc:])

error_list = np.abs(mean_list - L/2)

poly_s = np.polyfit(np.log10(dts), np.log10(error_list[0]), 1)
poly_l = np.polyfit(np.log10(dts), np.log10(error_list[1]), 1)

print(f"{poly_s[0] = }")
print(f"{poly_l[0] = }")

plt.scatter(dts, error_list[0], c='r', label = "Simple")
plt.plot(dts, np.power(10, poly_s[1]) * np.power(dts, poly_s[0]), 'r', label = f"$\sim\Delta t^{{{poly_s[0]:.2f}}}$")

plt.scatter(dts, error_list[1], c='b', label = "Lépingle")
plt.plot(dts, np.power(10, poly_l[1]) * np.power(dts, poly_l[0]), 'b', label = f"$\sim\Delta t^{{{poly_l[0]:.2f}}}$")

plt.grid(color="grey", linestyle='--')

plt.xticks(dts)

plt.xscale('log')
plt.yscale('log')

plt.legend()

plt.savefig("error.png")
