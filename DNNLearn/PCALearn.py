# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:13:48 2021

@author: thoma
"""
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------

import sys
sys.path.append(lib_path)
import pandas as pd
import numpy as np
import AZ_utilities as AZU
import AZ_analysis as AZA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FPS=120
# load single dictionary and tracking (e.g.)
IndDic_EC=np.load('D:/Movies/DataForAdam/IndDictionaries/EC_B2/210304_EmxGFP_Ctrl_B2_0_ANALYSIS.npy',allow_pickle=True).item()
fx_EC,fy_EC,_,_,_,_,_,ort_EC,_=AZU.grabTrackingFromFile('D:/Movies/DataForAdam/DataForAdam/ctrlB2/120Hz/Tracking/210304_EmxGFP_Ctrl_B2_0_tracking.npz',sf=0,ef=-1)

IndDic_EA=np.load('D:/Movies/DataForAdam/IndDictionaries/EA_B2/210309_EmxGFP_Asp_B2_1_ANALYSIS.npy',allow_pickle=True).item()
fx_EA,fy_EA,_,_,_,_,_,ort_EA,_=AZU.grabTrackingFromFile('D:/Movies/DataForAdam/DataForAdam/aspB2/120Hz/Tracking/210309_EmxGFP_Asp_B2_1_tracking.npz',sf=0,ef=-1)

# Grab boutStarts from Dictionary
starts_EC=IndDic_EC['data']['boutStarts']
starts_EA=IndDic_EC['data']['boutStarts']

# Grab fx, fy and ort for each bout
def cutCentreRotateTraj(fx,fy,ort,starts):
    ends=[]
    for i in starts[0:-2]:
        ends.append(i+120)
    starts=starts[0:-2]
    ## grab all real traces for each bout in mm with their origin subtracted
    _,_,trajHeadings,trajX,trajY=AZA.extractTrajFromStim(starts,ends,fx,fy,ort)
    
    # now rotate the origin subtracted trajectories to have a common starting heading
    rotTrajX,rotTrajY=AZA.rotateTrajectoriesByHeadings(trajX,trajY,trajHeadings)
    
    return rotTrajX,rotTrajY

TrajX_EC,TrajY_EC=cutCentreRotateTraj(fx_EC,fy_EC,ort_EC,starts_EC)
TrajX_EA,TrajY_EA=cutCentreRotateTraj(fx_EA,fy_EA,ort_EA,starts_EA)

# scale data to max (or STD...?)

#TrajY_EC_sc = StandardScaler().fit_transform(TrajY_EC)
#TrajX_EC_sc = StandardScaler().fit_transform(TrajX_EC)
#TrajY_EA_sc = StandardScaler().fit_transform(TrajY_EA)
#TrajX_EA_sc = StandardScaler().fit_transform(TrajX_EA)

TrajX_EC_sc=TrajX_EC/np.max(TrajY_EC)
TrajY_EC_sc=TrajY_EC/np.max(TrajY_EC)
TrajX_EA_sc=TrajX_EA/np.max(TrajX_EA)
TrajY_EA_sc=TrajY_EA/np.max(TrajY_EA)

TrajX_EC_sc=TrajX_EC_sc-np.min(TrajX_EC_sc)
TrajY_EC_sc=TrajY_EC_sc-np.min(TrajY_EC_sc)
TrajX_EA_sc=TrajX_EA_sc-np.min(TrajX_EA_sc)
TrajY_EA_sc=TrajY_EA_sc-np.min(TrajY_EA_sc)

#TrajX_EC_sc=(TrajX_EC-np.mean(TrajX_EC,axis=0))/np.max(TrajX_EC,axis=0)
#TrajY_EA_sc=(TrajY_EA-np.mean(TrajY_EA,axis=0))/np.max(TrajY_EA,axis=0)
#TrajX_EA_sc=(TrajX_EA-np.mean(TrajX_EA,axis=0))/np.max(TrajX_EA,axis=0)

# concatenate lists with list comprehension
Traj_EC = [i + j for i, j in zip(TrajX_EC_sc, TrajY_EC_sc)]
Traj_EA = [i + j for i, j in zip(TrajX_EA_sc, TrajY_EA_sc)]

pca_EC=PCA()
pca_EA=PCA()
pca_EC.fit(Traj_EC)
pca_EA.fit(Traj_EA)

cumsum_EC = np.cumsum(pca_EC.explained_variance_ratio_)
cumsum_EA = np.cumsum(pca_EA.explained_variance_ratio_)

#X_EC=pca_EC.fit_transform(Traj_EC)
#X_EA=pca_EA.fit_transform(Traj_EA)

## Convert metrics we want for analysis into a pandas dataframe
#df_EC = pd.DataFrame (IndDic_EC, columns = ['boutDists','boutAngles'])
#df_EA = pd.DataFrame (IndDic_EA, columns = ['boutDists','boutAngles'])
#
## Standardise (Zero Mean Unit Variance)
#xC = df_EC.loc[:, ['boutDists','boutAngles']].values
#xA = df_EC.loc[:, ['boutDists','boutAngles']].values

