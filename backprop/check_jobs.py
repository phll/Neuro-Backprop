import subprocess
import numpy as np
import fcntl
import os

# check on jobs. How many succeeded? How many failed?

name = "yinyang_pyralnet_vary_llag_55ms_reset_deltas"
tmp = "runs/"+name+"/tmp/"
results = "runs/"+name+"/results/"
config = "runs/"+name+"/config/"
user = "hd_fy440"
cores_per_job = 20


#check which jobs are currently active, eligible or blocked
try:
    res = subprocess.check_output('showq -u %s | grep -Po "^[0-9]+(?=\s+%s)"'%(user, user), shell=True).decode("utf-8")  #might contain canceling jobs as well!
    listed_jobs = [int(s) for s in res.split('\n')[:-1]]
except subprocess.CalledProcessError:
    listed_jobs = []

#load nemo-ids of all submitted jobs
job_nemo_ids = np.loadtxt(tmp + "job_nemo_ids", skiprows=1, dtype=np.int)

#'missing' jobs (might be finished successfully or might have failed)
cancelled_jobs = [ job[0] for job in job_nemo_ids if job[1] not in listed_jobs]
print("cancelled jobs: ", cancelled_jobs, len(cancelled_jobs))


#check which jobs failed (A job might not fail completely part only partially (ie some of its subprocesses))
restart_jobs = []
n_success = 0
n_part = 0
n_failed = 0
rebundle = []
do_rebundling = False
print("checking for failed jobs... (do not interrupt)\nDo you want to rebundle partially failed jobs? (y/n) ", end='')
while True:
    x = input()
    if x=="y":
        do_rebundling = True
        break
    elif x=="n":
        break

# are subprocesses found in results -> successfull
with open(results + "results.txt", "r+") as f_results:
    fcntl.flock(f_results, fcntl.LOCK_EX)
    file_content = f_results.read() # whole content of the results file

    for job in cancelled_jobs:
        print("check job %d... "%(job), end='')
        # load names of all processes of job from job-file
        f = open(tmp + str(job)+".job", "r")
        f.readline()
        run_names = []
        for conf_f in f:
            run_names.append(conf_f.split("/")[-1].replace(".conf\n", ''))
        f.close()

        #check if all finished successfully
        failed = []
        for rn in run_names:
            if rn not in file_content:
                failed += [rn]

        if len(failed) == 0:
            print("ok.")
            n_success += 1
        elif len(failed) == len(run_names):
            print("all failed -> restart")
            restart_jobs.append(job)
            n_failed += 1
        else:
            print("partially failed -> rebundle")
            n_failed += 1
            n_part += 1
            for fn in failed:
                print("----run %s failed."%(fn))
                if do_rebundling:
                    os.system("sed -i '/%s/d' %s"%(fn, tmp + str(job)+".job"))
            rebundle += failed

    fcntl.flock(f_results, fcntl.LOCK_UN)

running_jobs = [ job[0] for job in job_nemo_ids if job[1] in listed_jobs]
print("%d of %d jobs successfully completed"%(n_success, len(job_nemo_ids)))
print("%d of %d jobs running or waiting"%(len(running_jobs), len(job_nemo_ids)))
print("%d (%d -> %d runs) of %d jobs (partially) failed"%(n_failed, n_part, len(rebundle), len(job_nemo_ids)))


# rebundle partially failed jobs
if do_rebundling:
    n_new_jobs = int(np.ceil(len(rebundle)/cores_per_job))
    id_off = np.max(job_nemo_ids[:,0]) + 1
    job_nemo_ids = np.resize(job_nemo_ids, (len(job_nemo_ids) + n_new_jobs, 2))

    print("rebundle %d failed runs (from partially failed jobs) into %d new jobs"%(len(rebundle), n_new_jobs))

    # bundle runs into jobs (each job has 'cores_per_job' runs/subprocesses)
    for i in range(n_new_jobs):
        job_id = i + id_off
        # create a job-file containing all the config-files for the subprocesses it hosts
        f = open(tmp + "%d.job"%(job_id), "x")
        f.write(os.getcwd() + "/" + results + "\n")
        for k in range(i*cores_per_job, min((i+1)*cores_per_job, len(rebundle))):
            f.write(os.getcwd() + "/" + config + rebundle[k] + ".conf\n")
        f.close()
        restart_jobs += [job_id]
        job_nemo_ids[job_id, 0] = job_id

print("restarting jobs: ", restart_jobs, len(restart_jobs))

# restart jobs
for job in restart_jobs:
    result = subprocess.check_output('msub -N %s_%d_%d -l nodes=1:ppn=20,walltime=28:00:00,pmem=6GB job_pyral.sh "%s"' % (
    name, job + 1, len(job_nemo_ids), os.getcwd() + "/" + tmp + "%d.job" % (job)), shell=True)
    n_id = result.decode('utf-8').replace('\n', '')
    print(n_id)
    job_nemo_ids[job] = n_id

# save new nemo ids
f_ids = open(tmp + "job_nemo_ids", "w")
f_ids.write("job_id\t\tnemo_id\n")
for i in range(len(job_nemo_ids)):
    f_ids.write("%d\t\t%s\n" % (i, job_nemo_ids[i, 1]))
f_ids.close()
