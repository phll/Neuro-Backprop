import numpy as np
import matplotlib.pyplot as plt

##### Vary Learning-Lag #####

pyral_perf = np.loadtxt("runs/yinyang_pyralnet_vary_llag/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem, size]

plt.title("Yinyang PyraLNet performance")
plt.errorbar(pyral_perf[:,0], pyral_perf[:, 1]*100, yerr=pyral_perf[:, 2]*100, fmt='o')
plt.xlabel("learning-lag / ms")
plt.ylabel("accuracy / %")
plt.savefig("eval/yinyang_acc_vs_llag.png")
plt.clf()
