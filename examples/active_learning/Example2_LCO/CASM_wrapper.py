#!/usr/bin/env python

import sys, os

import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile
from mechanoChemML.workflows.active_learning.slurm_manager import numCurrentJobs, submitJob, waitForAll, checkPending


def submitCASM(N_jobs,phi,kappa,T,rnd,casm_project_dir='.',test=False):

    n = len(kappa)
    phi = np.array(n*[phi])
    T = np.array(n*[[T]])

    filename = os.path.dirname(__file__)+'/monte_settings.json.tmpl'
    #if test:
    #    filename = os.path.dirname(__file__)+'/monte_settings_test.json.tmpl'
    with open(filename,'r') as tmplFile:

        tmpl = json.load(tmplFile)
        for job in range(N_jobs):
            shutil.rmtree('job_{}'.format(job+1),ignore_errors=True)
            os.mkdir('job_{}'.format(job+1))

            inputF = copy.deepcopy(tmpl)

            for i in range(job,len(kappa),N_jobs):
                phiA = {}
                kappaA = {}
                for j in range(len(kappa[0])):
                    phiA[str(j)] = float(phi[i,j])
                    kappaA[str(j)] = float(kappa[i,j])
                inputF['driver']['custom_conditions']+=[{'tolerance': 0.001,
                                                         'temperature': 300.0,
                                                         'bias_phi': phiA,
                                                         'bias_kappa': kappaA,
                                                         'param_chem_pot': {'a': 0}}]

            with open('job_{0}/monte_settings_{0}.json'.format(job+1),'w') as outFile:
                json.dump(inputF,outFile,indent=4)

    command = ['cd job_$SLURM_ARRAY_TASK_ID',
               'casm --path {} monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json'.format(casm_project_dir)]
    if test:
        command = ['cd job_$SLURM_ARRAY_TASK_ID',
                   f'python {os.path.dirname(__file__)}/CASM_surrogate.py monte_settings_$SLURM_ARRAY_TASK_ID.json']        
    specs = {'job_name':'CASM2',
             'array': '1-{}'.format(N_jobs),
             'account': 'TG-MCH200011',
             'walltime': '2:00:00',
             'total_memory':'3G',
             'output_folder':'outputFiles'}
    submitJob(command,specs)

    sleep(30)
    while ( checkPending('CASM2') or (numCurrentJobs('CASM2') > 2)):
        sleep(15)

    
def compileCASMOutput(rnd):
    kappa = []
    eta = []
    phi = []
    T = []
    os.mkdir(f'round_{rnd}')
    for dir in os.listdir('.'):
        if 'job' in dir:
            if os.path.exists(dir+'/results.json'):
                with open(dir+'/results.json','r') as file:
                    data = json.load(file)
                    kappa += np.array([data['Bias_kappa({})'.format(i)] for i in range(7)]).T.tolist()
                    eta += np.array([data['<order_param({})>'.format(i)] for i in range(7)]).T.tolist()
                    phi += np.array([data['Bias_phi({})'.format(i)] for i in range(7)]).T.tolist()
                    T += np.array([data['T']]).T.tolist()
            shutil.move(dir,f'round_{rnd}')
                    
    kappa = np.array(kappa)
    eta = np.array(eta)
    phi = np.array(phi)
    T = np.array(T)
    mu = -2.*phi*(eta - kappa)
    dataOut = np.hstack((kappa,eta,phi,T,mu))
    dataOut = dataOut[~np.isnan(dataOut).any(axis=1)] #remove any rows with nan
    outVars = ['kappa','eta','phi']
    header = ''
    for outVar in outVars:
        for i in range(7):
            header += outVar+'_'+str(i)+' '
    header += 'T '
    for i in range(7):
        header += 'mu_'+str(i)+' '
    np.savetxt('data/results{}.txt'.format(rnd),
               dataOut,
               fmt='%.12f',
               header=header)
    if rnd==0:
        copyfile('data/results{}.txt'.format(rnd),'data/allResults{}.txt'.format(rnd))
    else:
        allResults = np.loadtxt('data/allResults{}.txt'.format(rnd-1))
        allResults = np.vstack((allResults,dataOut))
        np.savetxt('data/allResults{}.txt'.format(rnd),
                   allResults,
                   fmt='%.12f',
                   header=header)

def loadCASMOutput(rnd,singleRnd=False):

    dim = 7
    if singleRnd:
        kappa = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,:dim]
        eta = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,dim:2*dim]
        mu = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-dim:]
    else:
        kappa = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,:dim]
        eta = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,dim:2*dim]
        mu = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,-dim:]

    return kappa, eta, mu
