import numpy as np
from numpy import random as rand
import os

#Constants of diffusivity
K0 = 2 * 10**(-4)
K1 = 2 * 10**(-3)
alpha = 0.5

#Variable diffusivity function
def K_func(z):
    return K0 + z * K1 * np.exp(-z * alpha)

#Derivtive of variable diffusivity function
def K_func_prime(z):
    return K1 * np.exp(-z * alpha) * (1 - z * alpha)

#Constant diffusivity function
def K_const(z):
    return K1

#Derivative of constant diffusivity function
def K_const_prime(z):
    return 0

#Function for performing a single random step on all particles
def random_step(z, dt, a, b, L, simple = True):
    dW = rand.normal(0, np.sqrt(dt), len(z))
    z_temp = z + a(z) * dt + b(z) * dW
    #Simple reflection scheme
    if simple:
        z_temp = np.where(z_temp < 0, -z_temp, z_temp)
        z_temp = np.where(z_temp > L, 2 * L - z_temp, z_temp)
    #Lépingle reflection
    else:
        Vn = rand.exponential(2 * dt, len(z))
        Yn = 1/2 * (-a(z) * dt - b(z) * dW + np.sqrt(b(z)**2 * Vn + (-a(z) * dt - b(z) * dW)**2))
        z_temp = np.where(Yn - z >= 0, z_temp + Yn - z, z_temp)
        z_temp = np.where(z_temp > L, 2 * L - z_temp, z_temp)
    #Check that all particles are within permitted range
    assert (z_temp.all() >= 0)
    assert (z_temp.all() <= L)
    return z_temp


#Function to preform a number of random steps on all particles
def random_walk(Np, T, dt, K, K_prime, L, simple = True, save_interval = 1, rank = 0, full=True, savedir="noslurm"):
    next_save = save_interval
    Nt = int(np.floor(T / dt))
    save_num = int(np.floor(T / save_interval))

    save_index = 0
    z = rand.uniform(0, L, Np)
    if full:
        Zs = np.zeros((save_num, Np))
        Zs[0] = z
    else:
        Zs = z

    a = lambda z : K_prime(z)
    b = lambda z : np.sqrt(2 * K(z))

    for i in range(Nt - 1):
        z = random_step(z, dt, a, b, L, simple)
        if i*dt >= next_save:
            if full:
                Zs[save_index] = z
            else:
                save_name = "step_result_" + str(save_index) + "_" + str(rank)
                if simple:
                    save_name += "_s_"
                else:
                    save_name += "_l_"
                np.save(os.path.join(savedir, save_name), z)
                Zs = z
            next_save += save_interval
            save_index += 1
    return Zs, save_index
