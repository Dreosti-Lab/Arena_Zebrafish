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
tracking_paths_controls += glob.glob(DATAROOT + r"/GroupedTracking/EC_M0/*tracking.npz")
tracking_paths_controls_B0 = []
tracking_paths_controls_B0 += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*tracking.npz")

# Get tracking files (ablation)
tracking_paths_lesions = []
tracking_paths_lesions += glob.glob(DATAROOT + r"/GroupedTracking/EA_M0/*tracking.npz")
tracking_paths_lesions_B0 = []
tracking_paths_lesions_B0 += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*tracking.npz")

# Parameters
FPS=120

# Model Bouts
groups = [tracking_paths_controls, tracking_paths_lesions,tracking_paths_controls_B0,tracking_paths_lesions_B0]
group_bouts=[]
i=0
for group in groups:
    boutsS=[]
    j=0
    for tracking_path in group:
        print('load tracking ' + str(j+1) + ' of ' + str(len(group)) + ' in this group ' + str(i+1) + ' of ' + str(len(groups)))
        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Analyze bouts
#        print('Analysing')
        bouts = ARK_bouts.analyze(tracking)
#        bouts = ARK_bouts.filterTinyBouts(bouts)
        num_bouts = bouts.shape[0]
        
        # Plot tracking...
#        plt.subplot(1,2,1)
#        plt.plot(tracking[:, 2], tracking[:, 3], 'k.', MarkerSize=1, alpha=0.5)
#        plt.subplot(1,2,2)
#        plt.plot(tracking[:, 4], tracking[:, 7], 'k.', MarkerSize=1, alpha=0.5)
#        plt.xlim(-180, 180)
#        plt.ylim(-20, 50)
        
#        print('Analysed bouts')
        boutpathD,name=tracking_path.rsplit('\\',maxsplit=1)
        boutpath=boutpathD + '/' + name[0:-13] + '_bouts.npz'
        np.savez(boutpath,bouts=bouts)
        j+=1
#    group_bouts.append(boutsS)
    i+=1   
    
        
        