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
def random_step(ZV, dt, a, b, L, tau, simple = True):
    z = ZV[0] # Unpack position
    v = ZV[1] # Unpack velocity

    gamma = np.sqrt(b(z)**2 / tau**2) # Calculate gamma from the relation tau² * gamma² = 2 * K(z)

    dW = rand.normal(0, np.sqrt(dt), len(v)) # Stochastic variable
    v_temp = v - (1 / tau) * v * dt + gamma * dW # Calculate new velocity
    z += v * dt # Update position
    v = v_temp # Update velocity
 
    #Simple reflection scheme
    if simple:
        z = np.where(z < 0, -z, z)
        z = np.where(z > L, 2 * L - z, z)
    #Lépingle reflection
    else:
        Vn = rand.exponential(2 * dt, len(z))
        Yn = 1/2 * (-a(z) * dt - b(z) * dW + np.sqrt(b(z)**2 * Vn + (-a(z) * dt - b(z) * dW)**2))
        z = np.where(Yn - z >= 0, z + Yn - z, z)
        z = np.where(z > L, 2 * L - z, z)

    v = np.where(z < 0, -v, v)
    v = np.where(z > L, -v, v)

    #Check that all particles are within permitted range
    assert (z.all() >= 0)
    assert (z.all() <= L)
    return np.concatenate(([z], [v]), axis=0)


#Function to preform a number of random steps on all particles
def random_flight(Np, T, dt, K, K_prime, L, tau, simple = True, save_interval = 1, rank = 0, savedir="noslurm"):
    next_save = save_interval
    Nt = int(np.floor(T / dt))
    save_num = int(np.floor(T / save_interval))

    save_index = 0
    # Initialize position
    z = rand.uniform(0, L, Np)
    # Initialize velocity
    v = rand.uniform(0, 1, Np)

    ZV = np.concatenate(([z], [v]), axis=0) 

    print(f"{np.shape(ZV)}")

    a = lambda z : K_prime(z)
    b = lambda z : np.sqrt(2 * K(z))

    for i in range(Nt - 1):
        ZV = random_step(ZV, dt, a, b, L, tau, simple)
        if i*dt >= next_save:
            save_name = "step_result_" + str(save_index) + "_" + str(rank)
            if simple:
                save_name += "_s_"
            else:
                save_name += "_l_"
            np.save(os.path.join(savedir, save_name), ZV)
            next_save += save_interval
            save_index += 1
    return ZV, save_index
