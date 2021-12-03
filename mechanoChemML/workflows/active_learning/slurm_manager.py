#!/usr/bin/env python

import numpy as np
from subprocess import check_output, STDOUT
import os
from time import sleep

def numCurrentJobs(name):
    try:
        num = len(check_output(['squeue','-n',name],stderr=STDOUT).decode("utf-8").split('\n'))-2
    except:
        num = 1
        print('Error with numCurrentJobs: ',check_output(['squeue','-n',name],stderr=STDOUT).decode("utf-8"))
    return num
#return len(check_output(['squeue','-n',name],stderr=STDOUT).decode("utf-8").split('\n'))-2

def checkPending(name):
    try:
        val = False
        jobs = check_output(['squeue','-n',name],stderr=STDOUT).decode("utf-8").split('\n')
        for job in jobs:
            if 'PD' in job:
                val = True
                break
    except:
        val = False
        print('Error with check pending: ',check_output(['squeue','-n',name],stderr=STDOUT).decode("utf-8"))
    return val


def submitJob(command,specs={},is_dnsml=False):

    # Default values for LSF job script
    default = {'wall_time':'12:00:00',
               'nodes':1,
               'ntasks-per-node':1,
               'total_memory':'1G',
               'job_name':'default',
               'output_folder':'.',
               'queue':'shared'}

    # Incoporate any changes to the defaults
    default.update(specs)

    # Write out job script
    with open('submit.slrm','w') as fout:
        fout.write('#!/bin/bash\n#\n')
        fout.write("#SBATCH -t {}                  # wall time\n".format(default['wall_time']))
        if 'account' in default:
            fout.write("#SBATCH -A {}                  # account\n".format(default['account']))
        fout.write("#SBATCH --nodes {}                  \n".format(default['nodes']))
        fout.write("#SBATCH --ntasks-per-node={}                  \n".format(default['ntasks-per-node']))
        fout.write("#SBATCH --mem={}\n".format(default['total_memory']))
        fout.write('#SBATCH -J {}    # job name\n'.format(default["job_name"]))
        if 'gpu' in default['queue']:
            fout.write('#SBATCH --gpus=1')
        if 'array' in default:
            fout.write("#SBATCH --array={}                  # job array\n".format(default['array']))
        fout.write('#SBATCH -e {}/errors.%J       # error file name in which %J is replaced by the job ID \n'.format(default["output_folder"]))
        fout.write('#SBATCH -o {}/output.%J       # output file name in which %J is replaced by the job ID\n'.format(default["output_folder"]))
        fout.write('#SBATCH -p {}                       # choose the queue (partition) to use\n\n'.format(default["queue"]))
        fout.write('#SBATCH --export=ALL\n\n')
        
        if isinstance(command, list):
            for item in command:
                fout.write(item)
                fout.write('\n')
        else:
            fout.write(command)

    # Submit script
    os.system('sbatch submit.slrm')
        
def waitForAll(name,interval=15):
    sleep(2*interval)
    while ( numCurrentJobs(name) > 0 or checkPending(name) ):
        sleep(interval)
