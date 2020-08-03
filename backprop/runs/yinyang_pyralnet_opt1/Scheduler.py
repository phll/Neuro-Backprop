import json
from pathlib import Path
import numpy as np
from shutil import copyfile
import subprocess
import os

nodes_per_job = 1
cores_per_job = 20

name = "yinyang_pyralnet_opt1"
config = "runs/"+name+"/config/" # params for each execution
results = "runs/"+name+"/results/" # results will go here
tmp = "runs/"+name+"/tmp/" #job files (config files per job)

# prepare file structure

Path(config).mkdir(parents=True, exist_ok=True)
os.system('rm -rf %s*'%(config))

Path(results).mkdir(exist_ok=True)
os.system('rm -rf %s*'%(results))

Path(tmp).mkdir(exist_ok=True)
os.system('rm -rf %s*'%(tmp))
f = open(results+"results.txt", "x")
f.write("id\t\t\tlast val\t\tval acc\t\ttest acc\n")
f.close()

os.system('cp %s %s'%(__file__, "runs/"+name+"/Scheduler.py")) #copy this script to run directory


# build run configs and store them in 'config'
runs = []
gas = np.linspace(0.4, 0.8, 5)
gsoms = np.linspace(0.1, 0.8, 8)
l_1s = np.logspace(np.log10(5), -2, 20)
l_2_muls = np.logspace(-2, -4.5, 20)
ip_muls = [2]
seeds = [0]#np.random.ranndint(5000, 4)

print("g : ", gas)
print("l_1 : ", l_1s)
print("l_2_mul : ", l_2_muls)
print("ip_muls : ", ip_muls)

for ga in gas:
    for gsom in gsoms:
        if gsom>ga:
            continue
        for l_1 in l_1s:
            for l_2_mul in l_2_muls:
                for ip_mul in ip_muls:
                    run_name = "%.1f_%0.1f_%0.2e_%0.2e_%.1f"%(ga, gsom, l_1, l_2_mul*l_1, ip_mul)

                    params = {"name": run_name, "seed": seeds[0], "init_sps": True, "N_train": 6000, "N_test": 600,
                              "N_val": 600, "N_epochs": 45, "val_len": 40, "vals_per_epoch": 3,
                              "model": {"dims": [4, 120, 3], "act": "sigmoid", "dt": 0.1, "gl": 0.1, "gb": 1.0,
                                        "ga": ga, "gd": 1.0,
                                        "gsom": gsom,
                                        "eta": {"up": [l_1, l_1*l_2_mul], "pi": [0, 0],
                                                "ip": [ip_mul*l_1*l_2_mul, 0]},
                                        "bias": {"on": True, "val": 0.5},
                                        "init_weights": {"up": 0.1, "down": 1, "pi": 1, "ip": 0.1}, "tau_w": 30, "noise": 0,
                                        "t_pattern": 100,
                                        "out_lag": 80, "tau_0": 3, "learning_lag": 0}}

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
    result = subprocess.check_output('msub -N %s_%d_%d -l nodes=1:ppn=20,walltime=28:00:00,pmem=6GB job.sh "%s"'%(name, i+1, jobs, os.getcwd() + "/" + tmp + "%d.job"%(i)), shell=True)
    n_id = result.decode('utf-8').replace('\n', '')
    print(n_id)
    f_ids.write("%d\t\t%s\n"%(i, n_id))

f_ids.close()
