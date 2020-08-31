import numpy as np
import matplotlib.pyplot as plt

DO_VARY_LLAG_PLOT = False
DO_LLAG_DEV_TRACES = True

##### Vary Learning-Lag #####
if DO_VARY_LLAG_PLOT:
    pyral_perf = np.loadtxt("./runs/yinyang_pyralnet_vary_llag/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_200 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_200ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50_2 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms_2/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50_3 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms_3/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50_4 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms_4/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_300 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_300ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)[:60]# [learning_lag, mean, std, sem]


    plt.title("Yinyang PyraLNet performance")
    #plt.errorbar(pyral_perf[:,0], pyral_perf[:, 1]*100, yerr=pyral_perf[:, 2]*100, fmt='o', label="100 ms", markersize=4)
    #plt.errorbar(pyral_perf_200[:,0], pyral_perf_200[:, 1]*100, yerr=pyral_perf_200[:, 2]*100, fmt='^', label="200 ms", markersize=4)
    plt.errorbar(pyral_perf_50[:,0], pyral_perf_50[:, 1]*100, yerr=pyral_perf_50[:, 2]*100, fmt='s', label="50 ms", markersize=4)
    #plt.errorbar(pyral_perf_50_2[:,0], pyral_perf_50_2[:, 1]*100, yerr=pyral_perf_50_2[:, 2]*100, fmt='s', label="50 ms _2", markersize=4)
    plt.errorbar(pyral_perf_50_3[:,0], pyral_perf_50_3[:, 1]*100, yerr=pyral_perf_50_3[:, 2]*100, fmt='s', label="50 ms _3", markersize=4)
    plt.errorbar(pyral_perf_50_4[:,0], pyral_perf_50_4[:, 1]*100, yerr=pyral_perf_50_4[:, 2]*100, fmt='s', label="50 ms _4", markersize=4)

    #plt.errorbar(pyral_perf_300[:,0], pyral_perf_300[:, 1]*100, yerr=pyral_perf_300[:, 2]*100, fmt='*', label="300 ms", markersize=4, c="purple")
    plt.legend()
    plt.xlabel("learning-lag / ms")
    plt.ylabel("accuracy / %")
    plt.savefig("eval/yinyang_acc_vs_llag.png")
    plt.clf()


if DO_LLAG_DEV_TRACES:
