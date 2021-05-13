DATAROOT = '/home/kampff/Data/Arena'
LIBROOT = '/home/kampff/Repos/dreosti-lab/Arena_Zebrafish'

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
import ARK_utilities
import ARK_bouts

# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# Get tracking files (controls)
tracking_paths_controls = []
tracking_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/EC_M0/*.npz")
#tracking_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*.npz")

# Get tracking files (ablation)
tracking_paths_lesions = []
tracking_paths_lesions += glob.glob(DATAROOT + "/GroupedTracking/EA_M0/*.npz")
#tracking_paths_lesions += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*.npz")

# Parameters
FPS=120

# Model IBI
groups = [tracking_paths_controls, tracking_paths_lesions]
group_var_true = []
group_var_random = []
for group in groups:
    var_true = []
    var_random = []
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Analyze bouts
        bouts = ARK_bouts.analyze(tracking)
        num_bouts = bouts.shape[0]

        # Extract bout params
        dSpace = bouts[:,5]
        dAngle = bouts[:,4]

        # Shuffle bout params
        suffled_indices = np.random.permutation(np.arange(num_bouts))
        dSpace_shuffle = dSpace[suffled_indices]
        dAngle_shuffle = dAngle[suffled_indices]

        # Measure local bout variability
        window_length = 25
        num_measurements = num_bouts - window_length
        variance_true = 0
        variance_random = 0
        for i in range(num_measurements):
            variance_true += np.std(dSpace[i:(i+window_length)])
            variance_random += np.std(dSpace_shuffle[i:(i+window_length)])
        
        # Store mean absolute error (VAR)
        var_true.append(variance_true / num_measurements)
        var_random.append(variance_random / num_measurements)
    
    # Store groups VAR
    group_var_true.append(var_true)
    group_var_random.append(var_random)

# Report
VAR_relative_controls = np.array(group_var_true[0])/np.array(group_var_random[0])
VAR_relative_lesions = np.array(group_var_true[1])/np.array(group_var_random[1])
VAR_absolute_controls = np.array(group_var_true[0])
VAR_absolute_lesions = np.array(group_var_true[1])
print("Controls vs Lesions (relative): {0} v {1}".format(np.mean(VAR_relative_controls), np.mean(VAR_relative_lesions)))
print("Controls vs Lesions (absolute): {0} v {1}".format(np.mean(VAR_absolute_controls), np.mean(VAR_absolute_lesions)))

#FIN