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
group_mae_true = []
group_mae_random = []
for group in groups:
    mae_true = []
    mae_random = []
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Analyze bouts
        bouts = ARK_bouts.analyze(tracking)
        num_bouts = bouts.shape[0]

        # Measure inter-bout intervals
        ibis = np.diff(bouts[:,1])
        num_ibis = len(ibis)

        # Predict next IBI from previous
        future_steps = 1
        num_predictions = num_bouts - future_steps - 1
        error_true = 0
        error_random = 0
        for i in range(num_predictions):
            error_true += np.abs((ibis[i] - ibis[i+future_steps]))
            error_random += np.abs(ibis[i] - ibis[np.random.randint(0, num_ibis)])
        
        # Store mean absolute error (MAE)
        mae_true.append(error_true / num_predictions)
        mae_random.append(error_random / num_predictions)
    
    # Store groups MAEs
    group_mae_true.append(mae_true)
    group_mae_random.append(mae_random)

# Report
MAE_relative_controls = np.array(group_mae_true[0])/np.array(group_mae_random[0])
MAE_relative_lesions = np.array(group_mae_true[1])/np.array(group_mae_random[1])
print("Controls: {0}".format(np.mean(MAE_relative_controls)))
print("Lesions: {0}".format(np.mean(MAE_relative_lesions)))

#FIN