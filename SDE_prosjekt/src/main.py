from random_walk import *
from random_flight import random_flight
from pathlib import Path
import sys, os, argparse

parser = argparse.ArgumentParser() #Be able to use flags for inputs

parser.add_argument("-r", "--reflection", help="Reflection scheme (s)imple or (l)epingle. default is s.", default="s")
parser.add_argument("-dt", "--timestep", help="Size of timestep. Default is 50s.", default="50")
parser.add_argument("-np", "--particles", help="Number of particles. Default is 1e4.", default="1e4")
parser.add_argument("-t", "--time", help="Total time of simulation in hours. Default is 6h.", default="6")
parser.add_argument("-l", "--depth", help="Depth of water in meters. Default is 2m.", default="2")
parser.add_argument("-s", "--scheme", help="Choose between random [flight] or [walk]. Default is walk.", default="walk")
parser.add_argument("-T", "--tau", help="Characteristic time scale of particle velocity in seconds. Default is 2s.", default="2")

args = parser.parse_args()

reflection = args.reflection
timestep = args.timestep
particles = args.particles
time = args.time
depth = args.depth
scheme = args.scheme
tau = args.tau

simple=True

#Parse reflection scheme
if reflection == "s" or reflection == "simple":
    simple = True
elif reflection == "l" or reflection == "lepingle":
    simple = False
else:
    print("Invalid reflection scheme", reflection)

#Parse timestep
try:
    dt = int(timestep)
except:
    print("Invalid timestep", timestep)

#Parse number of particles
try:
    Np = int(float(particles))
except:
    print("Invalid number of particles", particles)

#Parse total times
try:
    T = float(time) * 3600
except:
    print("Invalid time", time)

#Parse depth
try:
    L = float(depth)
except:
    print("Ivalid depth", depth)

if scheme == "walk":
    res = random_walk(Np, T, dt, K_func, K_func_prime, L, simple)
elif scheme == "flight":
    try:
        tau = float(tau)
    except:
        print("Invalid tau", tau)
    res = random_flight(Np, T, dt, K_func, K_func_prime, L, tau, simple)
else:
    print("Invalid scheme", scheme)
filename = f"results_random_{scheme}_{reflection}_{timestep}_{particles}_{time}_{depth}_.npy"
np.save(filename, res)
