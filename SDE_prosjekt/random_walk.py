import numpy as np
from numpy import random as rand

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
        Vn = rand.exponential(2*dt, len(z))
        Yn = 1/2 * (-a(z) * dt - b(z) * dW + np.sqrt(b(z)**2 * Vn + (-a(z) * dt - b(z) * dW)**2))
        z_temp = np.where(Yn - z >= 0, z_temp + Yn - z, z_temp)
        z_temp = np.where(z_temp > L, 2 * L - z_temp, z_temp)
    #Check that all particles are within permitted range
    assert (z_temp.all() >= 0)
    assert (z_temp.all() <= L)
    return z_temp


#Function to preform a number of random steps on all particles
def random_walk(Np, T, dt, K, K_prime, L, simple = True):
    Nt = int(np.floor(T / dt))

    print(f"{Nt = }, {Np = }")
    Zs = np.zeros((Nt, Np))
    z0 = rand.uniform(0, L, Np)
    Zs[0] = z0

    a = lambda z : K_prime(z)
    b = lambda z : sqrt(2 * K(z))

    for i in range(Nt - 1):
        Zs[i+1] = random_step(Zs[i], dt, a, b, L, simple)
    return Zs


"""
Np = int(1e4) #Number of particles
T = 6 * 3600 #Total time
L = 2 #Total depth
bins = 200 #Number of bins in histogram

#Random walk performed with steplenght dt = 50
dt = 50
Z50 = random_walk(Np, T, dt, K_func, K_func_prime, L, False)

hist50 = np.histogram(Z50, bins, (0, L))
depth = hist50[1][:-1]
C50 = hist50[0] / (np.sum(hist50[0]) * (depth[1] - depth[0]))


#Random walk performed with steplenght dt = 10
dt = 10
Z10 = random_walk(Np, T, dt, K_func, K_func_prime, L, False)

hist10 = np.histogram(Z10, bins, (0, L))
C10 = hist10[0] / (np.sum(hist10[0]) * (depth[1] - depth[0]))

#Ploting
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
fig.suptitle(f"Results of random walk; {Np = }, T = {T / 3600}h")

plt.ylim(2.1, -0.1)

axs[0].plot(K_func(depth), depth)
axs[0].set_xlim(0.0, 0.0025)
axs[0].set_xlabel("Diffusivity")
axs[0].set_ylabel("Depth (m)")
axs[0].grid()

axs[1].plot(C50, depth, label=f"$\Delta t = 50s$")
axs[1].plot(C10, depth, label=f"$\Delta t = 10s$")
axs[1].set_xlim(0.4, 0.6)
axs[1].set_xlabel("Concentration")
axs[1].grid()
axs[1].legend()

plt.savefig("plot.png")
"""