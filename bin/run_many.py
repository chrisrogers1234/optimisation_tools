import json
import time
import os
import sys
import subprocess
import glob
import atexit
import signal

from optimisation_tools.utils import utilities

PROC_QUEUE = []
PROC_RUNNING = []
UNIQUE_ID = 0
N_PROCS = 11
TARGET_SCRIPT = "run_many_"
TIME_0 = time.time()
LOG_DIR = "logs"

def do_at_exit():
    proc_queue = poll()
    if is_scarf():
        print("Killing all child processes ... good bye")
        for jobid, name in proc_queue:
            subprocess.check_output(['scancel', jobid])
            print("   ", jobid)
        proc_queue = poll()
        print(len(proc_queue), "processes left")
    else:
        pgid = os.getpgid(os.getpid())
        print("Killing all child processes of process group", pgid, "... good bye")
        os.killpg(pgid, signal.SIGKILL)
    print("Au revoir")

def will_make_new_procs(temp_proc_queue):
    global PROC_QUEUE, N_PROCS
    return len(temp_proc_queue) < N_PROCS and len(PROC_QUEUE) > 0

def archive_logs():
    old_dir = os.path.join(LOG_DIR, "old")
    if not os.path.isdir(old_dir):
        print(f"{old_dir} is not a directory; not archiving log files")
        return
    log_list = glob.glob(os.path.join(f"{LOG_DIR}", "*.log"))
    print(f"Archiving {len(log_list)} log files to {old_dir}")
    for fname in log_list:
        new_name = os.path.split(fname)[1]
        new_name = os.path.join(old_dir, new_name)
        os.rename(fname, new_name)

def poll_process_queue():
    global UNIQUE_ID, PROC_RUNNING, PROC_QUEUE, TIME, LOG_DIR
    print("\r", round(time.time()-TIME_0, 1), "...",
          "Running", len(PROC_RUNNING), 
          "with", len(PROC_QUEUE), "queued", end=" ")
    temp_proc_queue = poll()
    if will_make_new_procs(temp_proc_queue):
        print()
    while will_make_new_procs(temp_proc_queue):
        subproc_args, logname = PROC_QUEUE.pop(0)
        UNIQUE_ID += 1
        job_log = LOG_DIR+"/"+logname+".log"
        job_name = f"run_many_{UNIQUE_ID}"
        if is_scarf():
            subproc_args = ["salloc", "--job-name", job_name, "-N1", "srun", "-n1", "-t5", "-o", job_log, "-e", job_log, "--job-name", job_name]+subproc_args
            srun_log = f"logs/{job_name}.log"
            logfile = open(srun_log, "w")
        else:
            logfile = open(job_log, "w")
        proc = subprocess.Popen(subproc_args, stdout=logfile, stderr=subprocess.STDOUT)
        temp_proc_queue.append((proc, job_name))
        print("Running", subproc_args, "with log", job_log)
        time.sleep(1)
    PROC_RUNNING = temp_proc_queue

def poll():
    if is_scarf():
        temp_proc_queue = poll_scarf()
    else:
        temp_proc_queue = poll_laptop()
    return temp_proc_queue

def poll_laptop():
    temp_proc_queue = []
    for proc, jobname in PROC_RUNNING:
        if proc.poll() == None:
            temp_proc_queue.append((proc, jobname))
        else:
            print("\nPID", proc.pid, "finished with return code", proc.returncode)
    sys.stdout.flush()
    return temp_proc_queue

def poll_scarf():
    global TARGET_SCRIPT
    output = subprocess.check_output(['squeue', '--me', '-O', 'JobID,Name'])
    output = output.decode('utf-8')
    lines = output.split('\n')
    temp_proc_queue = []
    for line in lines:
        if TARGET_SCRIPT not in line:
            continue
        words = line.split()
        if len(words) != 2:
            raise RuntimeError(f"could not parse line {line}")
        proc = words[0]
        jobname = words[1]
        temp_proc_queue.append( (proc, jobname) )
    for proc, jobname in PROC_RUNNING:
        if proc != None and proc not in [item[0] for item in temp_proc_queue]:
            print("\nJobId", proc, "finished")
    return temp_proc_queue

def is_scarf():
    uname = str(subprocess.check_output(['uname', '-a']))
    return uname.find('scarf.rl.ac.uk') > -1

def load_configs():
    config_file_name, config = utilities.get_config("run_many/")
    print("Loading configs from ", config_file_name)
    job_list = [[str(shell_arg) for shell_arg in job] for job in config.get_job()]
    return job_list

def main(config_file):
    atexit.register(do_at_exit)
    archive_logs()
    configs = load_configs()
    if os.getenv("OPAL_EXE_PATH") == None:
        raise ValueError("No OPAL_EXE_PATH set")
    global N_PROCS, TARGET_SCRIPT, UNIQUE_ID
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if is_scarf():
        N_PROCS = 20
    for config in configs:
        print("Setting config", config)
        log_file = config[0].split("/")[-1]
        log_file = log_file[:-3]
        if len(config) > 0:
            log_file = log_file+"_"+"_".join(config[1:])
        run_one = os.path.expandvars("${OPTIMISATION_TOOLS}/bin/run_one.py")
        proc_tuple = (["python", run_one]+config, log_file)
        PROC_QUEUE.append(proc_tuple)
    print(len(PROC_QUEUE), "jobs")
    while len(PROC_QUEUE) > 0 or len(PROC_RUNNING) > 0:
        poll_process_queue()
        if len(PROC_QUEUE) == 0 and len(PROC_RUNNING) == 0:
            break
        time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ${OPTIMISATION_TOOLS}/bin/run_many.py <job_file>")
        sys.exit(1)
    main(sys.argv[1])
    print("\nFinished")



