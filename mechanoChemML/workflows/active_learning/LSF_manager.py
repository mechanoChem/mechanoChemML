#!/usr/bin/env python

import numpy as np
from subprocess import check_output, STDOUT
import os
from time import sleep

def numCurrentJobs(name):
    return len(check_output(['bjobs','-J',name],stderr=STDOUT).decode("utf-8").split('\n'))-2

def submitJob(command,specs={},is_dnsml=False):

    # Default values for LSF job script
    default = {'wall_time':'12:00',
               'n_processes':1,
               'total_memory':5000,
               'mem_per_process':1000,
               'job_name':'default',
               'output_folder':'.',
               'queue':'normal'}

    # Incoporate any changes to the defaults
    default.update(specs)

    # Write out job script
    with open('submit.lsf','w') as fout:
        fout.write('#!/bin/bash\n#\n')
        if is_dnsml:
            fout.write('#BSUB -a tbb')
        fout.write("#BSUB -W {}                  # wall time\n".format(default['wall_time']))
        fout.write("#BSUB -n {}                  # n processes\n".format(default['n_processes']))
        fout.write("#BSUB -R rusage[mem={}]       # amount of total memory in MB for all processes\n".format(default['total_memory']))
        fout.write('#BSUB -R "span[ptile={}]"        # number of processes per host\n'.format(min(20,default["n_processes"])))
        #fout.write('#BSUB -R "affinity[thread(1):cpubind=thread:distribute=pack]"   # thread(t) means "t" threads per tasks\n')
        fout.write('#BSUB -R "affinity[core(1):cpubind=core:distribute=balance]" # bind 1 core per process\n')
        fout.write('#BSUB -M {}                   # amount of memory in MB per process\n'.format(default["mem_per_process"]))
        fout.write('#BSUB -J "{}"    # job name\n'.format(default["job_name"]))
        fout.write('#BSUB -e {}/errors.%J       # error file name in which %J is replaced by the job ID \n'.format(default["output_folder"]))
        fout.write('#BSUB -o {}/output.%J       # output file name in which %J is replaced by the job ID\n'.format(default["output_folder"]))
        fout.write('#BSUB -q {}                       # choose the queue to use: gpu_p100, normal or large_memory\n\n'.format(default["queue"]))

        if isinstance(command, list):
            for item in command:
                fout.write(item)
                fout.write('\n')
        else:
            fout.write(command)

    # Submit script
    os.system('bsub < submit.lsf')
        
def waitForAll(name,interval=15):
    sleep(2*interval)
    while ( numCurrentJobs(name) > 0):
        sleep(interval)
