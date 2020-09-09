import json
import pandas as pd

### builds pandas dataframe from results.txt file

name = "yinyang_pyralnet_vary_tau_0_100ms_llag_15ms"
results = "runs/"+name+"/results/"
configs = "runs/"+name+"/config/"
COLLAPSE_SEEDS = True
n_seeds = 10

rows = []

with open(results + "results.txt", "r") as f_res:
    f_res.readline() # skip header
    for l in f_res:
        l = l.split('\t')
        conf = l[0] + ".conf" # config file
        run_id = l[0].split("__")[-1]
        test_acc = float(l[-1].replace('\n', '')) # test accuracy

        with open(configs + conf) as json_file:
            params = json.load(json_file)

            l1 = params["model"]["eta"]["up"][0]
            l2 = params["model"]["eta"]["up"][1]
            l2_mul = l2 / l1
            ga = params["model"]["ga"]
            gsom = params["model"]["gsom"]
            ip1 = params["model"]["eta"]["ip"][0]
            ip1_mul = ip1 / l2
            l_pi = params["model"]["eta"]["pi"][0]
            learning_lag = params["model"]["learning_lag"]
            tau_0 = params["model"]["tau_0"]
            tau_w = params["model"]["tau_w"]

            rows += [[run_id, ga, gsom, l1, l2, l2_mul, ip1, ip1_mul, l_pi, learning_lag, tau_0, tau_w, test_acc]]

df = pd.DataFrame(rows, columns = ["run_id", "ga", "gsom", "l1", "l2", "l2_mul", "ip1", "ip1_mul", "l_pi", "learning_lag", "tau_0", "tau_w", "test accuracy"])
df = df.sort_values(by=["test accuracy"], ascending=False)

df.to_csv (results + "results_df.csv", index = False, header=True)

print(df[:50])
print(df)

if COLLAPSE_SEEDS:
    df_stats = df.groupby(by=["ga", "gsom", "l1", "l2", "l2_mul", "ip1", "ip1_mul", "l_pi", "learning_lag", "tau_0", "tau_w"]).agg(["mean", "std", "sem", "min", "max", "size"])

    print(df_stats[:50])
    print(df_stats)
    print(df_stats.sort_values(by=("test accuracy", "mean"), ascending=False))

    df_stats.reset_index().to_csv(results + "results_df_stats.csv", index=False, header=True)
