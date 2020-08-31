import numpy as np
import matplotlib.pyplot as plt
import PyraLNet as pyral
import Dataset
import os
import json

DO_VARY_LLAG_PLOT = True
DO_LLAG_DEV_TRACES = False

##### Vary Learning-Lag #####
if DO_VARY_LLAG_PLOT:
    pyral_perf = np.loadtxt("./runs/yinyang_pyralnet_vary_llag/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_200 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_200ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_50_2 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_50ms_2/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)# [learning_lag, mean, std, sem]
    pyral_perf_300 = np.loadtxt("./runs/yinyang_pyralnet_vary_llag_300ms/results/results_df_stats.csv", usecols=[8,9,10,11], delimiter=",", skiprows=2)[:60]# [learning_lag, mean, std, sem]


    plt.title("Yinyang PyraLNet performance")
    plt.errorbar(pyral_perf[:,0], pyral_perf[:, 1]*100, yerr=pyral_perf[:, 2]*100, fmt='o', label="100 ms", markersize=4)
    plt.errorbar(pyral_perf_200[:,0], pyral_perf_200[:, 1]*100, yerr=pyral_perf_200[:, 2]*100, fmt='^', label="200 ms", markersize=4)
    plt.errorbar(pyral_perf_50[:,0], pyral_perf_50[:, 1]*100, yerr=pyral_perf_50[:, 2]*100, fmt='s', label="50 ms", markersize=4)
    #plt.errorbar(pyral_perf_50_2[:,0], pyral_perf_50_2[:, 1]*100, yerr=pyral_perf_50_2[:, 2]*100, fmt='s', label="50 ms _2", markersize=4)
    plt.errorbar(pyral_perf_300[:,0], pyral_perf_300[:, 1]*100, yerr=pyral_perf_300[:, 2]*100, fmt='*', label="300 ms", markersize=4, c="purple")

    plt.legend()
    plt.xlabel("learning-lag / ms")
    plt.ylabel("accuracy / %")
    plt.savefig("eval/yinyang_acc_vs_llag.png")
    plt.clf()


#### Deviations between soma rates and predicted rates ####
if DO_LLAG_DEV_TRACES:
    # load meta data
    base = "runs/yinyang_pyralnet_llag_weight_breadcrumbs/"
    runs_dict = {}
    params = None
    seeds = []
    for f in os.listdir(base + "config/"):
        if f.endswith(".conf"):
            with open(base + "config/" + f) as json_file:
                f_params = json.load(json_file)
                llag = f_params["model"]["learning_lag"]
                name = f_params["name"]
                seed = f_params["seed"]
                seeds += [seed]
                if llag in runs_dict:
                    runs_dict[llag][seed] = name
                else:
                    runs_dict[llag] = {seed: name}
                del f_params["name"]
                del f_params["model"]["learning_lag"]
                del f_params["seed"]
                if params is None:
                    params = f_params
                elif params != f_params:
                    print("parameter mismatch for %s: . Abort."%(f))
                    print(params, f_params)
                    exit(0)
    seeds = np.unique(seeds)
    breadcrumbs = params["breadcrumbs"]
    del params["breadcrumbs"]
    llags = np.sort(list(runs_dict.keys()))
    print(llags, seeds, breadcrumbs)

    N_patterns = 10
    plt.figure(figsize=(22, 12))
    for i, ll in enumerate(llags):
        for k, bc_idx in enumerate(breadcrumbs):
            X, Y = Dataset.YinYangDataset(size=100, flipped_coords=True, seed=40)[:N_patterns]
            target_seq = np.ones((N_patterns, 3), dtype=pyral.dtype) * 0.1
            target_seq[:, 1 * Y] = 1.0
            params["model"]["learning_lag"] = ll
            act = pyral.sigmoid
            net = pyral.Net(params["model"], act=act)
            net.load_weights(base + "results/weights_epoch_%d_%s.npy"%(bc_idx, runs_dict[ll][seeds[0]]))

            rec_pots = [["pyr_soma", "pyr_basal", "inn_soma", "inn_dendrite", "Delta_up"], ["pyr_soma", "pyr_basal"]]
            records, T, r_in, u_trgt, out_seq = net.run(X, np.hstack((target_seq, np.ones((N_patterns,1)))), rec_pots=rec_pots, rec_dt=0.1)
            T = T.reshape(N_patterns, -1)[0]

            gl, gb, gd, ga = params["model"]["gl"], params["model"]["gb"], params["model"]["gd"], params["model"]["ga"]
            dev_hid_pyr = np.abs(act(records[0]["pyr_soma"].data) - act(gb/(ga+gl+gb)*records[0]["pyr_basal"].data)).reshape(N_patterns, -1, net.dims[1])
            dev_hid_inn = np.abs(act(records[0]["inn_soma"].data) - act(gd/(gl+gd)*records[0]["inn_dendrite"].data)).reshape(N_patterns, -1, net.dims[2])
            dev_out_pyr = np.abs(act(records[1]["pyr_soma"].data) - act(gb/(gl+gb)*records[1]["pyr_basal"].data)).reshape(N_patterns, -1, net.dims[2])

            delta_hid_up = np.abs(records[0]["Delta_up"].data).reshape(N_patterns, -1, net.dims[1], net.dims[0] + params["model"]["bias"]["on"])

            tr_hid_pyr = np.mean(dev_hid_pyr , axis=(0, 2))
            tr_out_pyr = np.mean(dev_out_pyr, axis=(0, 2))
            tr_hid_inn = np.mean(dev_hid_inn, axis=(0, 2))

            plt.subplot(len(llags), len(breadcrumbs), i * len(breadcrumbs) + k + 1)
            if i==0:
                plt.title("After %d epochs"%(bc_idx))
            elif i==len(llags)-1:
                plt.xlabel("time / ms")
            if k==0:
                plt.ylabel("Learning-lag %d ms"%(ll))
            plt.plot(T, tr_hid_pyr, label="hidden pyr")
            plt.plot([ll, ll], [10**-15, 10], c="r", ls="--")
            #plt.plot(T, tr_out_pyr, label="hidden inn")
            #plt.plot(T, tr_hid_inn, label="out pyr")
            plt.ylim([0.000001, np.max([tr_hid_pyr])*2])
            plt.yscale("log")

            ax2 = plt.gca().twinx()
            ax2.plot(T, np.mean(delta_hid_up, axis=(0, 2, 3)), c="green")
            if k == len(breadcrumbs)-1:
                ax2.set_ylabel("Delta W_up")
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.tight_layout()
    plt.savefig("eval/traces.png")
    plt.show()
