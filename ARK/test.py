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
import ARK_bouts

# Reload libraries
import importlib
importlib.reload(ARK_bouts)

# Get tracking files (controls)
tracking_paths_controls = []
tracking_paths_controls += glob.glob(DATAROOT + "/ctrlB2/120Hz/Tracking/*.npz")
tracking_paths_controls += glob.glob(DATAROOT + "/ctrlB2/100Hz/Tracking/*.npz")

# Get tracking files (ablation)
tracking_paths_lesions = []
tracking_paths_lesions += glob.glob(DATAROOT + "/aspB2/120Hz/Tracking/*.npz")
tracking_paths_lesions += glob.glob(DATAROOT + "/aspB2/100Hz/Tracking/*.npz")

# Load all controls
groups = [tracking_paths_controls, tracking_paths_lesions]
mean_bs_groups = []
for group in groups:
    mean_bs = []
    for tracking_path in group:
    
        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']

        ## Display Track 
        #plt.plot(tracking[:,0], tracking[:,1], 'r')
        #plt.plot(tracking[:,2], tracking[:,3], 'b')
        #plt.plot(tracking[:,4], tracking[:,5], 'g')
        #plt.show()

        # Characterize Bouts
        bouts = ARK_bouts.analyze(tracking)
        num_bouts = bouts.shape[0]
        if(num_bouts < 300):
            continue
        median_angle = np.median(np.abs(bouts[:,4]))
        median_distance = np.median(np.abs(bouts[:,5]))
        #print(median_distance)
        median_angle = 10.0
        median_distance = 0.2

        # Measure correlation between subsequent bouts
        N = 100
        bout_similarities = np.zeros((num_bouts-N-1, N))
        for b in range(num_bouts - N - 1):
            this_angle = bouts[b,4]
            this_distance = bouts[b,5]
            for n in range(N):
                delta_angle = (this_angle - bouts[b+n+1, 4])
                delta_distance = (this_distance - bouts[b+n+1, 5])
                delta_angle = 0
                bout_similarities[b, n] = np.sqrt((delta_angle/median_angle)**2 + (delta_distance/median_distance)**2)
        mean_bout_similarities = np.mean(bout_similarities, axis=0)
        mean_bs.append(mean_bout_similarities)
        #mean_bs.append(mean_bout_similarities/mean_bout_similarities[0])
        #plt.plot(mean_bout_similarities)
        #plt.show()
    mean_bs_groups.append(np.array(mean_bs))

# Predict position and orientation from previous 5 seconds
# - Input: X second snippet
# - Output: prediction at some point in the future (X seconds?) 

controls = mean_bs_groups[0]
lesions = mean_bs_groups[1]

# Plot
plt.plot(controls.T, 'b', alpha=0.2)
plt.plot(lesions.T, 'r', alpha=0.2)
plt.plot(np.mean(controls, axis=0), 'b')
plt.plot(np.mean(lesions, axis=0), 'r')
plt.show()

#FIN