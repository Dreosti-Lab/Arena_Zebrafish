# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:22:06 2021

@author: thoma
"""

DATAROOT = r'D:\Movies/DataForAdam/DataForAdam/'
LIBROOT = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish'

# Set library paths
import os
import sys
lib_path = LIBROOT + "/ARK/libs"
ARK_lib_path = LIBROOT + "/libs"
sys.path.append(lib_path)
sys.path.append(ARK_lib_path)

# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ARK_utilities
import ARK_bouts
import AZ_utilities as AZU
# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# Get tracking files (controls)
tracking_paths_controls = []
tracking_paths_controls += glob.glob(DATAROOT + r"/GroupedTracking/EC_M0/*bouts.npz")
tracking_paths_controls_B0 = []
tracking_paths_controls_B0 += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*bouts.npz")

# Get tracking files (ablation)
tracking_paths_lesions = []
tracking_paths_lesions += glob.glob(DATAROOT + r"/GroupedTracking/EA_M0/*bouts.npz")
tracking_paths_lesions_B0 = []
tracking_paths_lesions_B0 += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*bouts.npz")






# Parameters
FPS=120

# Model Bouts
groups = [tracking_paths_controls, tracking_paths_lesions] # tracking_paths_controls_B0,tracking_paths_lesions_B0]
group_mean_acf=[]
group_SEM_acf=[]
SeUp=[]
SeDown=[]
group_mean_pturn=[]
group_SEM_pturn=[]
group_bouts=[]
i=0
plt.figure()

for group in groups:
    acfS=[]
    pturnS=[]
    boutsS=[]
    j=0
    for bout_path in group:
        # Load bouts (starts, peaks, stops, durations, dAngle, dSpace, x, y)
        bouts = np.load(bout_path)['bouts']
        num_bouts = bouts.shape[0]

        # Extract bout angles
        dAngle = bouts[:,4]

        # Discretise angle vector and project across time
        discTurns,pturn = AZU.discretiseAngleVector(dAngle)
        starts=bouts[:,0]
        discTimeTurns=np.zeros(np.int(np.max(starts)))
        boutCounter=0
        state=0
        for k in range(np.int(np.max(starts))):
            if k == starts[boutCounter]:
                state=discTurns[boutCounter]
                boutCounter+=1
            discTimeTurns[k]=state
                        
        
        # take the autocorrelation function of discretised vector for this fish
        #collect autocorrelations
        TimeSecs=(5*60)
        acfS.append(sm.tsa.acf(discTimeTurns,nlags=(TimeSecs*120)-1,fft=True))
    

        pturnS.append(pturn)
                
        
        j+=1
    group_mean_acf.append(np.mean(acfS,axis=0))
    group_SEM_acf.append(np.divide(np.std(acfS,axis=0),np.sqrt(len(acfS))))
    SeUp.append(group_mean_acf[i] + group_SEM_acf[i])
    SeDown.append(group_mean_acf[i] - group_SEM_acf[i])
    
    group_mean_pturn.append(np.mean(pturnS,axis=0))
    group_SEM_pturn.append(np.divide(np.std(pturnS,axis=0),np.sqrt(len(pturnS))))        
#    group_bouts.append(boutsS)
    
    i+=1   

colors=['Green','Magenta']
labels=['Control','Lesion']
xTime=np.linspace(0,(TimeSecs),(TimeSecs*FPS))

plt.figure('ACFM0')

for j,thisLabel in enumerate(labels):
    plt.plot(xTime,group_mean_acf[j],label=thisLabel,color=colors[j])
    plt.plot(xTime,SeUp[j],color=colors[j],alpha=0.5)
    plt.plot(xTime,SeDown[j],color=colors[j],alpha=0.5)
    plt.fill_between(xTime, SeUp[j], y2=SeDown[j],  color=colors[j],alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Auto-correlation coefficient')
    plt.legend()

    
        