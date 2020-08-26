import numpy as np
import matplotlib.pyplot as plt

##### Vary Learning-Lag #####

pyral_perf = np.loadtxt("runs/yinyang_pyralnet_vary_llag/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
pyral_perf_200 = np.loadtxt("runs/yinyang_pyralnet_vary_llag_200ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]

plt.title("Yinyang PyraLNet performance")
plt.errorbar(pyral_perf[:,0], pyral_perf[:, 1]*100, yerr=pyral_perf[:, 2]*100, fmt='o', label="100 ms")
plt.errorbar(pyral_perf_200[:,0], pyral_perf_200[:, 1]*100, yerr=pyral_perf_200[:, 2]*100, fmt='^', label="200 ms", c="g")
plt.legend()
plt.xlabel("learning-lag / ms")
plt.ylabel("accuracy / %")
plt.savefig("eval/yinyang_acc_vs_llag.png")
plt.clf()
