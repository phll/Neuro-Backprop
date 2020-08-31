import numpy as np
import matplotlib.pyplot as plt
import PyraLNet as pyral
import Dataset

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


#### Deviations between soma rates and predicted rates ####
if DO_LLAG_DEV_TRACES:
    N_patterns = 10
    ga = 0.28
    gsom = 0.39
    gl = 0.1
    gb, gd = 1.0, 1.0
    X, Y = Dataset.YinYangDataset(size=100, flipped_coords=True, seed=40)[:N_patterns]
    target_seq = np.ones((N_patterns, 3), dtype=pyral.dtype) * 0.1
    target_seq[:, 1 * Y] = 1.0
    params = {"dims": [4, 120, 3], "dt": 0.1, "gl": gl, "gb": gb, "ga": ga, "gd": gd,
              "gsom": gsom,
              "eta": {"up": [6.1, 0.00012], "pi": [0, 0], "ip": [0.00024, 0]},
              "bias": {"on": True, "val": 0.5},
              "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 0.1}, "tau_w": 30, "noise": 0, "t_pattern": 50,
              "out_lag": 40, "tau_0": 3, "learning_lag": 0}
    act = pyral.sigmoid
    net = pyral.Net(params, act=act)
    net.reflect()

    rec_pots = [["pyr_soma", "pyr_basal", "inn_soma", "inn_dendrite"], ["pyr_soma", "pyr_basal"]]
    records, T, r_in, u_trgt, out_seq = net.run(X, np.hstack((target_seq, np.ones((N_patterns,1)))), rec_pots=rec_pots, rec_dt=0.1)
    T = T.reshape(N_patterns, -1)[0]

    dev_hid_pyr = np.abs(act(records[0]["pyr_soma"].data) - gb/(ga+gl+gb)*act(records[0]["pyr_basal"].data)).reshape(N_patterns, -1, net.dims[1])
    dev_hid_inn = np.abs(act(records[0]["inn_soma"].data) - gd/(gl+gd)*act(records[0]["inn_dendrite"].data)).reshape(N_patterns, -1, net.dims[2])
    dev_out_pyr = np.abs(act(records[1]["pyr_soma"].data) - gb/(gl+gb)*act(records[1]["pyr_basal"].data)).reshape(N_patterns, -1, net.dims[2])

    tr_hid_pyr = np.mean(dev_hid_pyr , axis=(0, 2))
    tr_out_pyr = np.mean(dev_out_pyr, axis=(0, 2))
    tr_hid_inn = np.mean(dev_hid_inn, axis=(0, 2))
    plt.plot(T, tr_hid_pyr, label="hidden pyr")
    plt.plot(T, tr_out_pyr, label="hidden inn")
    plt.plot(T, tr_hid_inn, label="out pyr")
    plt.ylim([0, np.max([tr_hid_inn, tr_hid_pyr, tr_out_pyr])*1.1])
    plt.legend()
    plt.show()