import numpy as np
import os
from scipy.signal import correlate
import matplotlib.pyplot as plt

load_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "data/results_s_16_100000_6.0_2_.npy"))

res = np.load(load_path)

res_mean = np.mean(res, axis = 1) - np.mean(res)

res_correlation = np.correlate(res_mean, res_mean, mode='full')

plt.plot(res_correlation)
plt.show()
