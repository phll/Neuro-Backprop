import subprocess
import numpy as np
import fcntl
import os

name = "bars_pyralnet_bias_off"
tmp = "runs/"+name+"/tmp/"
results = "runs/"+name+"/results/"
user = "hd_fy440"

#check which jobs are currently active, eligible or blocked
try:
    res = subprocess.check_output('showq -u %s | grep -Po "^[0-9]+(?=\s+%s)"'%(user, user), shell=True).decode("utf-8")
    listed_jobs = [int(s) for s in res.split('\n')[:-1]]
except subprocess.CalledProcessError:
    listed_jobs = []

#load nemo-ids of all submitted jobs
job_nemo_ids = np.loadtxt(tmp + "job_nemo_ids", skiprows=1, dtype=np.int)

#'missing' jobs
cancelled_jobs = [ job[0] for job in job_nemo_ids if job[1] not in listed_jobs]
print("cancelled jobs: ", cancelled_jobs, len(cancelled_jobs))

#check which jobs failed
restart_jobs = []
delete_lines = []
with open(results + "results.txt", "r+") as f_results: #are processes found in results -> successfull
    fcntl.flock(f_results, fcntl.LOCK_EX)
    file_content = f_results.read()

    for job in cancelled_jobs:
        #load names of all processes of job from job-file
        f = open(tmp + str(job)+".job", "r")
        f.readline()
        run_names = []
        for conf_f in f:
            run_names.append(conf_f.split("/")[-1].replace(".conf\n", ''))
        f.close()

        #check if all finished successfully
        failed = False
        for rn in run_names:
            if rn not in file_content:
                failed = True
                break

        if failed:
            delete_lines += run_names
            restart_jobs.append(job)

    #delete partial results of failed jobs
    print("delete: ")
    f_results.seek(0)
    f_results.truncate()
    for l in file_content.split("\n")[:-1]:
        if l == '':
            continue
        skip = False
        for dl in delete_lines:
            if dl in l:
                skip = True
                print(l)
                break
        if not skip:
            f_results.write(l+"\n")
    fcntl.flock(f_results, fcntl.LOCK_UN)


running_jobs = [ job[0] for job in job_nemo_ids if job[1] in listed_jobs]
print("%d of %d jobs successfully completed"%(len(job_nemo_ids)-len(restart_jobs)-len(running_jobs), len(job_nemo_ids)))
print("%d of %d jobs running or waiting"%(len(running_jobs), len(job_nemo_ids)))
print("%d of %d jobs failed"%(len(restart_jobs), len(job_nemo_ids)))
print("restarting jobs: ", restart_jobs, len(restart_jobs))

#restart jobs
for job in restart_jobs:
    result = subprocess.check_output('msub -N %s_%d_%d -l nodes=1:ppn=20,walltime=28:00:00,pmem=6GB job.sh "%s"' % (
    name, job + 1, len(job_nemo_ids), os.getcwd() + "/" + tmp + "%d.job" % (job)), shell=True)
    n_id = result.decode('utf-8').replace('\n', '')
    print(n_id)
    job_nemo_ids[job] = n_id

#save new nemo ids
f_ids = open(tmp + "job_nemo_ids", "w")
f_ids.write("job_id\t\tnemo_id\n")
for i in range(len(job_nemo_ids)):
    f_ids.write("%d\t\t%s\n" % (i, job_nemo_ids[i, 1]))
f_ids.close()
