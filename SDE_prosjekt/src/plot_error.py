import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_error(error_data_file_name, error_plot_name, dts):
    error_data = np.load(error_data_file_name)
    plt.scatter(dts, error_data[0], label="Simple scheme")
    plt.scatter(dts, error_data[1], label="Lépingle scheme")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(error_plot_name)

error_data_file_name = sys.argv[1]
try:
    error_plot_name = sys.argv[2]
except:
    error_plot_name = "error_plot.png"

dts = np.array([2, 3, 4, 6, 8, 12, 16])

plot_error(error_data_file_name, error_plot_name, dts)
