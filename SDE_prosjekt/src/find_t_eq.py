from random_walk import random_walk, K_func, K_func_prime
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI


dt = 30
Np = int(1e6)
T = 3600 * 6
L = 2
save_interval =  #Save interval in seconds
num_saves = int(np.floor(T / ( save_interval)))

MPI.Init()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

my_Np = int(Np / size)

if rank == 0:
    Np = size * my_Np
    print(f"{size = }")

print(f"Rank {rank} is running random_walk with {my_Np = }")
my_res = random_walk(my_Np, T, dt, K_func, K_func_prime, L, True, save_interval)

comm.Barrier()

res = comm.gather(my_res, root=0)

if rank == 0:
    res = np.reshape(res, (num_saves, Np))
    filename = f"results_s_{dt}_{Np}_{T/3600}_{L}_.npy"
    np.save(filename, res)

    t = np.linspace(0, T, num_saves)
    res_mean = np.mean(res, axis=1)

    plt.plot(t / 3600, res_mean)
    title = f"{Np = }, T = {T/3600}h"
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("<z>")
    plt.savefig(f"mean_concentration_s_{dt}_{Np}_{T/3600}_{L}_.png")
    plt.show()

MPI.Finalize()
