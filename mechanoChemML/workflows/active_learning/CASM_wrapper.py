#!/usr/bin/env python

import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile

def submitCASM(N_jobs,mu_test,eta,rnd,casm_project_dir='.',test=False,job_manager='LSF'):

    # Calculate and write out the predicted kappa values to the CASM input files
    n = len(eta)
    phi = np.array(n*[[5.,3.,3.,3.]])
    kappa = eta + 0.5*mu_test/phi
    T = np.array(n*[[600.]])

    dataOut = np.hstack((kappa,eta,phi,T,mu_test))
    np.savetxt('data/CASMinput{}.txt'.format(rnd),
               dataOut,
               fmt='%.12f',
               header='kappa_0 kappa_1 kappa_2 kappa_3 eta_0 eta_1 eta_2 eta_3 phi_0 phi_1 phi_2 phi_3 T mu_0 mu_1 mu_2 mu_3')

    kappa = np.expand_dims(kappa,-1).tolist()

    with open(os.path.dirname(__file__)+'/monte_settings.json.tmpl','r') as tmplFile:

        tmpl = json.load(tmplFile)
        for job in range(N_jobs):
            shutil.rmtree('job_{}'.format(job+1),ignore_errors=True)
            os.mkdir('job_{}'.format(job+1))

            inputF = copy.deepcopy(tmpl)

            for i in range(job,len(kappa),N_jobs):
                inputF['driver']['conditions_list']+=[{'tolerance': 0.001,
                                                       'temperature': 600.0,
                                                       'phi': [[5.],[3.],[3.],[3.]],
                                                       'kappa': kappa[i]}]

            with open('job_{0}/monte_settings_{0}.json'.format(job+1),'w') as outFile:
                json.dump(inputF,outFile,indent=4)

    command = ['cwd=$PWD',
               'mv job_$LSB_JOBINDEX {}'.format(casm_project_dir),
               'cd {}/job_$LSB_JOBINDEX'.format(casm_project_dir),
               '$CASMPREFIX/bin/casm monte -s monte_settings_$LSB_JOBINDEX.json',
               'cd ../',
               'mv job_$LSB_JOBINDEX $cwd']
    if test:
        if job_manager == 'LSF':
            command = ['cd job_$LSB_JOBINDEX'.format(casm_project_dir),
                       'python -u {}/CASM_surrogate.py monte_settings_$LSB_JOBINDEX.json'.format(os.path.dirname(__file__)),
                       'cd ../'] 
        elif job_manager == 'slurm':
            command = ['cd job_$SLURM_ARRAY_TASK_ID'.format(casm_project_dir),
                       'python -u {}/CASM_surrogate.py monte_settings_$SLURM_ARRAY_TASK_ID.json'.format(os.path.dirname(__file__)),
                       'cd ../'] 

    if job_manager == 'LSF':
        from mechanoChemML.workflows.active_learning.LSF_manager import submitJob, waitForAll
        specs = {'job_name':'CASM_[1-{}]'.format(N_jobs),
                 'queue': 'gpu_p100',
                 'output_folder':'outputFiles'}
        name = 'CASM*'
    elif job_manager == 'slurm':
        from mechanoChemML.workflows.active_learning.slurm_manager import submitJob, waitForAll
        specs = {'job_name':'CASM',
                 'array': '1-{}'.format(N_jobs),
                 'account': 'TG-MCH200011',
                 'walltime': '2:00:00',
                 'total_memory':'3G',
                 'output_folder':'outputFiles'}
        name = 'CASM'
        
    submitJob(command,specs)

    waitForAll(name)

def compileCASMOutput(rnd):
    kappa = []
    eta = []
    phi = []
    T = []
    for dir in os.listdir('.'):
        if 'job' in dir:
            if os.path.exists(dir+'/results.json'):
                with open(dir+'/results.json','r') as file:
                    data = json.load(file)
                    kappa += np.array([data['kappa_{}'.format(i)] for i in range(4)]).T.tolist()
                    eta += np.array([data['<op_val({})>'.format(i)] for i in range(4)]).T.tolist()
                    phi += np.array([data['phi_{}'.format(i)] for i in range(4)]).T.tolist()
                    T += np.array([data['T']]).T.tolist()

    kappa = np.array(kappa)
    eta = np.array(eta)
    phi = np.array(phi)
    T = np.array(T)
    mu = -2.*phi*(eta - kappa)
    dataOut = np.hstack((kappa,eta,phi,T,mu))
    dataOut = dataOut[~np.isnan(dataOut).any(axis=1)] #remove any rows with nan
    np.savetxt('data/results{}.txt'.format(rnd),
               dataOut,
               fmt='%.12f',
               header='kappa_0 kappa_1 kappa_2 kappa_3 eta_0 eta_1 eta_2 eta_3 phi_0 phi_1 phi_2 phi_3 T mu_0 mu_1 mu_2 mu_3')
    if rnd==0:
        copyfile('data/results{}.txt'.format(rnd),'data/allResults{}.txt'.format(rnd))
    else:
        allResults = np.loadtxt('data/allResults{}.txt'.format(rnd-1))
        allResults = np.vstack((allResults,dataOut))
        np.savetxt('data/allResults{}.txt'.format(rnd),
                   allResults,
                   fmt='%.12f',
                   header='kappa_0 kappa_1 kappa_2 kappa_3 eta_0 eta_1 eta_2 eta_3 phi_0 phi_1 phi_2 phi_3 T mu_0 mu_1 mu_2 mu_3')

def loadCASMOutput(rnd,singleRnd=False):

    if singleRnd:
        dataIn = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,[4,5,6,7,13,14,15,16]]
    else:
        dataIn = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,[4,5,6,7,13,14,15,16]]
    features = dataIn[:,:4]
    labels = dataIn[:,4:]

    return features, labels
