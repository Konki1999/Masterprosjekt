import numpy as np
import os
import time
#import matplotlib.pyplot as plt
from random_walk import random_walk, K_func, K_func_prime
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

dts = np.array([2, 3, 4, 6, 8, 12, 16])
Np = int(1e5)
T = 3600 * 6
L = 2
t_eq = 3600 * 0.5 # Time from start of simulation to approximate equilibrium is reached
save_interval = 100 # Number of seconds between each save point
save_num = int(np.floor(T / save_interval)) # Number of timesteps that are saved

full = False # Save result to disk for each timestep instead of saving all in one file
slurm = ""
try:
    slurm = os.environ["SLURM_JOB_ID"]
except:
    slurm = "noslurm"

r_eq = (T - t_eq) / T # Factor with which to reduce the number of timesteps to include in the error analysis
saves_for_error_calc = int(np.floor(r_eq * save_num)) # Include saves_for_error_calc last save points in calculation of error

MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    mean_list = np.zeros((2,len(dts)))
    try:
        os.mkdir(slurm)
    except:
        rm_command = "rm -r " + slurm
        os.system(rm_command)
        os.mkdir(slurm)

comm.barrier()

N = Np // size # Divide particles into equal

for i, dt in enumerate(dts):
    savedir = os.path.join(slurm, str(dt))
    if rank == 0:
        try:
            os.mkdir(savedir)
        except:
            rm_command = "rm -r " + savedir
            os.system(rm_command)
            os.mkdir(savedir)
    comm.barrier()
    res_s, last_save_s = random_walk(N, T, dt, K_func, K_func_prime, L, True, save_interval, rank=rank, full=full, savedir=savedir)
    res_l, last_save_l = random_walk(N, T, dt, K_func, K_func_prime, L, False, save_interval, rank=rank, full=full, savedir=savedir)

    comm.barrier()
    if not full and rank == 0:
        res_s = np.zeros((save_num, size * N))
        res_l = np.zeros((save_num, size * N))
        for save_point in range(last_save_s):
            for r in range(size):
                filename_s = "step_result_" + str(save_point) + "_" + str(r) + "_s_.npy"
                res_s[save_point, r*N:(r+1)*N] = np.load(os.path.join(savedir, filename_s))

                filename_l = "step_result_" + str(save_point) + "_" + str(r) + "_l_.npy"
                res_l[save_point, r*N:(r+1)*N] = np.load(os.path.join(savedir, filename_l))

        # Take mean of position along all axis for the last saves_for_error_calc save points along time axis
        mean_list[0,i] = np.mean(res_s[-saves_for_error_calc:])
        mean_list[1,i] = np.mean(res_l[-saves_for_error_calc:])

if rank == 0:
    error_list = np.abs(mean_list - L/2)

    poly_s = np.polyfit(np.log10(dts), np.log10(error_list[0]), 1)
    poly_l = np.polyfit(np.log10(dts), np.log10(error_list[1]), 1)

    err = np.sqrt(1 / (L**2 / 12 * 10 * Np))

    print("poly_s[0] =", poly_s[0])
    print("poly_l[0] =", poly_l[0])

    np.save("error_data.npy", error_list)

MPI.Finalize()
