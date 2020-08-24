import json
from pathlib import Path
import numpy as np
import subprocess
import os
from sklearn.model_selection import ParameterSampler

nodes_per_job = 1
cores_per_job = 20

name = "yinyang_pyralnet_vary_llag_200ms"
config = "runs/"+name+"/config/" # params for each execution
results = "runs/"+name+"/results/" # results will go here
tmp = "runs/"+name+"/tmp/" #job files (config files per job)

# prepare file structure
print("Delete all existing files in %s. Continue? "%(name), end='')
_ = input()

Path(config).mkdir(parents=True, exist_ok=True)
os.system('rm -rf %s*'%(config))

Path(results).mkdir(exist_ok=True)
os.system('rm -rf %s*'%(results))

Path(tmp).mkdir(exist_ok=True)
os.system('rm -rf %s*'%(tmp))
f = open(results+"results.txt", "x")
f.write("id\t\t\tlast val\t\tval acc\t\ttest acc\n")
f.close()

#copy files for reproduction
os.system('cp %s %s'%(__file__, "runs/"+name+"/Scheduler.py"))
os.system('cp %s %s'%("PyraLNet.py", "runs/"+name+"/PyraLNet.py"))
os.system('cp %s %s'%("job_pyral.sh", "runs/"+name+"/job_pyral.sh"))

# build run configs and store them in 'config'
runs = []
seeds = [2304, 3446, 123, 4354, 8956, 283, 384, 78, 2, 6566]
run_id = 0

print("build config files")

for hp in [{"ga": 0.28, "gsom": 0.34, "l_1": 6.1, "l_2_mul": 0.00012, "ip_mul": 2.0}]:
    ga = hp["ga"]
    gsom = hp["gsom"]
    l_1 = hp["l_1"]
    l_2_mul = hp["l_2_mul"]
    ip_mul = hp["ip_mul"]

    for llag in np.linspace(0, 30, 60):
        for seed in seeds:
            run_name = "%.2f_%.2f_%.2e_%.2e_%.1f__%d"%(ga, gsom, l_1, l_2_mul*l_1, llag, run_id)
            run_id += 1

            params = {"name": run_name, "seed": seed, "init_sps": True, "track_sps": False, "N_train": 6000,
                      "N_test": 600, "N_val": 600, "N_epochs": 45, "val_len": 100, "vals_per_epoch": 1,
                      "model": {"dims": [4, 120, 3], "act": "sigmoid", "dt": 0.1, "gl": 0.1, "gb": 1.0,
                                "ga": ga, "gd": 1.0,
                                "gsom": gsom,
                                "eta": {"up": [l_1, l_1*l_2_mul], "pi": [0, 0],
                                        "ip": [ip_mul*l_1*l_2_mul, 0]},
                                "bias": {"on": True, "val": 0.5},
                                "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 0.1}, "tau_w": 30, "noise": 0,
                                "t_pattern": 200,
                                "out_lag": 80, "tau_0": 3, "learning_lag": llag}}

            with open('%s.conf'%(config+run_name), 'w') as file:
                file.write(json.dumps(params))

            runs +=[run_name]

jobs = int(np.ceil(len(runs)/cores_per_job))
print("submit %d jobs."%(jobs))

os.system('rm -rf %s*' % (tmp))

# keep a list of the nemo-ids
f_ids = open(tmp + "job_nemo_ids", "x")
f_ids.write("job_id\t\tnemo_id\n")

# bundle runs into jobs (each job has 'cores_per_job' runs/subprocesses)
for i in range(jobs):
    # create a job-file containing all the config-files for the subprocesses it hosts
    f = open(tmp + "%d.job"%(i), "x")
    f.write(os.getcwd() + "/" + results + "\n")
    for k in range(i*cores_per_job, min((i+1)*cores_per_job, len(runs))):
        f.write(os.getcwd() + "/" + config + runs[k] + ".conf\n")
    f.close()

    # submit job and save nemo-id
    result = subprocess.check_output('msub -N %s_%d_%d -l nodes=1:ppn=20,walltime=52:00:00,pmem=6GB job_pyral.sh "%s"'%(name, i+1, jobs, os.getcwd() + "/" + tmp + "%d.job"%(i)), shell=True)
    n_id = result.decode('utf-8').replace('\n', '')
    print(n_id)
    f_ids.write("%d\t\t%s\n"%(i, n_id))

f_ids.close()
