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
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
pre_window = 25
post_window = 75

# Bout PCA
groups = [tracking_paths_controls, tracking_paths_lesions]
group_bout_trajectories = []
for group in groups:
    bout_trajectories = []
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Extract tracking
        X = tracking[:,2]
        Y = tracking[:,3]
        A = tracking[:,7]

        # dX, dY, dA(ngle)
        dX = np.diff(X, prepend=X[0])
        dY = np.diff(Y, prepend=Y[0])
        dA = ARK_utilities.diffAngle(A)
        dA = ARK_utilities.filterTrackingFlips(dA) / (2*np.pi)

        # Analyze bouts
        bouts = ARK_bouts.analyze(tracking)
        num_bouts = bouts.shape[0]

        # Extract and align bout trajectories
        for bout in bouts:
            index = np.int(bout[0]) # Align to start
            if(index < pre_window):
                continue
            if(index > (num_frames-post_window)):
                continue
            tX = X[(index-pre_window):(index+post_window)]
            tY = Y[(index-pre_window):(index+post_window)]
            tdA = dA[(index-pre_window):(index+post_window)]
            tA = np.cumsum(tdA)
            align_ort = 2*np.pi*(A[index] / 360.0)

            # Center bout trajectory
            sX = tX - tX[25]
            sY = tY - tY[25]

            # Align bout trajectory
            aX = np.cos(align_ort) * (sX) - np.sin(align_ort) * (sY)
            aY = np.sin(align_ort) * (sX) + np.cos(align_ort) * (sY)
            aA = tA - tA[25]

            # Store aligned trjectory
            bout_trajectories.append(np.hstack((aX, aY, aA)))

    # Store groups
    group_bout_trajectories.append(np.array(bout_trajectories))

# Report
avg_bout_trajectory_Controls = np.mean(group_bout_trajectories[0], axis=0)
avg_bout_trajectory_Lesions = np.mean(group_bout_trajectories[1], axis=0)
std_bout_trajectory_Controls = np.std(group_bout_trajectories[0], axis=0)
std_bout_trajectory_Lesions = np.std(group_bout_trajectories[1], axis=0)

plt.plot(std_bout_trajectory_Controls)
plt.plot(std_bout_trajectory_Lesions)
plt.show()

plt.plot(avg_bout_trajectory_Controls)
plt.plot(avg_bout_trajectory_Lesions)
plt.show()

# Do PCA (on all bouts)
pca = PCA(n_components=20)
pca.fit(np.vstack((group_bout_trajectories[0], group_bout_trajectories[1])))

# Dim Reduction
controls_compressed = pca.transform(group_bout_trajectories[0])
lesions_compressed = pca.transform(group_bout_trajectories[1])

plt.plot(controls_compressed[:,2], controls_compressed[:,4], '.', alpha=0.01)
plt.plot(lesions_compressed[:,2], lesions_compressed[:,4], 'r.', alpha=0.01)
plt.show()

# Reconstrcut bouts
for b in range(10):
    plt.plot(group_bout_trajectories[0][b,:])
    recon = np.zeros(300)
    recon += pca.mean_
    for c in range(0,20):
        recon += (pca.components_[c] * controls_compressed[b][c])
    plt.plot(recon, 'r')
    plt.show()

# Report
plt.plot(pca_controls.explained_variance_ratio_[:20])
plt.plot(pca_lesions.explained_variance_ratio_[:20])
plt.show()

# Save
pca_path = DATAROOT + '/pca.npz'
np.savez(pca_path, pca_mean=pca.mean_, pca_components=pca.components_, controls_compressed=controls_compressed, lesions_compressed=lesions_compressed)

#FIN