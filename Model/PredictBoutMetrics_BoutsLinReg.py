# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:24:54 2021

@author: thoma
"""
# %% LOAD DATA, OPTIONS AND PREPARE VARIABLES
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
from sklearn.linear_model import LinearRegression

import ARK_utilities
import ARK_bouts
import AZ_utilities as AZU
# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# Get bout files (controls)
bout_paths_controls = []
bout_paths_controls += glob.glob(DATAROOT + r"/GroupedTracking/EC_M0/*bouts.npz")
bout_paths_controls_B0 = []
bout_paths_controls_B0 += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*bouts.npz")

# Get bout files (ablation)
bout_paths_lesions = []
bout_paths_lesions += glob.glob(DATAROOT + r"/GroupedTracking/EA_M0/*bouts.npz")
bout_paths_lesions_B0 = []
bout_paths_lesions_B0 += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*bouts.npz")

# Parameters
FPS=120
history_length = 5 # 
prediction_target = 1 # 
train_examples_per_fish = 1000
test_examples_per_fish = 200

# Set groups and empty results arrays
groups = [bout_paths_controls, bout_paths_lesions] # [bout_paths_controls_B0,bout_paths_lesions_B0]
train_set_control=[]   
train_goal_control=[]

train_set_lesion=[]
train_goal_lesion=[]

test_set_control=[]
test_goal_control=[]

test_set_lesion=[]
test_goal_lesion=[]

# %% Construct test and training sets for two groups
def buildTrainTestSets(group):
    num_fish=len(group)
    # Create train / test datasets
    num_train = train_examples_per_fish * num_fish
    num_test = test_examples_per_fish * num_fish

    train_set = np.zeros((num_train, history_length, 3))
    test_set = np.zeros((num_test, history_length, 3))

    train_goal = np.zeros((num_train, 3))
    test_goal = np.zeros((num_test, 3))
    
    train_counter=0
    test_counter=0
    for boutpath in group:
        # Load bouts (starts, peaks, stops, durations, dAngle, dSpace, x, y) 
        bouts=np.load(boutpath)['bouts']
        dAngle=bouts[0:-1,4]
        dSpace=bouts[0:-1,5]
        IBI=np.diff(bouts[:,0]) / FPS
        num_bouts=len(IBI)
        
        train_indices = np.random.randint(history_length, num_bouts-prediction_target, train_examples_per_fish)
        
        for i in train_indices:
            history_range = np.arange(i - history_length, i)
        
            train_set[train_counter,0] = np.mean(dSpace[history_range])
            train_set[train_counter,1] = np.mean(dAngle[history_range])
            train_set[train_counter,2] = np.mean(IBI[history_range])
            train_goal[train_counter, 0] = dSpace[i + prediction_target]
            train_goal[train_counter, 1] = dAngle[i + prediction_target]
            train_goal[train_counter, 2] = IBI[i + prediction_target]
        
        test_indices = np.random.randint(history_length, num_bouts-prediction_target, test_examples_per_fish)
        for ind,i in enumerate(test_indices):
            history_range = np.arange(i - history_length, i)
            
            test_set[test_counter,:,0] = dSpace[history_range]
            test_set[test_counter,:,1] = dAngle[history_range]
            test_set[test_counter,:,2] = IBI[history_range]
            test_goal[test_counter, 0] = dSpace[i + prediction_target]
            test_goal[test_counter, 1] = dAngle[i + prediction_target]
            test_goal[test_counter, 2] = IBI[i + prediction_target]
    
    return train_set,train_goal,test_set,test_goal

train_set_control,train_goal_control,test_set_control,test_goal_control = buildTrainTestSets(groups[0])
train_set_lesion,train_goal_lesion,test_set_lesion,test_goal_lesion = buildTrainTestSets(groups[1])
model_dSpace_control = LinearRegression()
model_dSpace_control.fit(train_set_control[:,0], train_goal_control[:,0])

