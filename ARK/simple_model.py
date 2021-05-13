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

# Reload libraries
import importlib
importlib.reload(ARK_utilities)

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
history_length = FPS*5
prediction_target = FPS*2

# Model
groups = [tracking_paths_controls, tracking_paths_lesions]
group_mae_Xs = []
group_mae_Ys = []
group_mae_As = []
for group in groups:
    mae_Xs = []
    mae_Ys = []
    mae_As = []
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Extract tracking
        X = tracking[:,2]
        Y = tracking[:,3]
        A = tracking[:,7]

        # Linear model - dX, dY, dA(ngle) continues into future
        dX = np.diff(X, prepend=X[0])
        dY = np.diff(Y, prepend=Y[0])
        dA = ARK_utilities.diffAngle(A)
        dA = ARK_utilities.filterTrackingFlips(dA) / (2*np.pi)

        # Build averaging kernel (history_length + history_length)
        kernel = np.hstack((np.ones(history_length), np.zeros(history_length)))/history_length

        # Convolve
        dX_smooth = np.convolve(dX, kernel, mode='same')
        dY_smooth = np.convolve(dY, kernel, mode='same')
        dA_smooth = np.convolve(dA, kernel, mode='same')

        # Prediction
        pX = X + dX_smooth*prediction_target
        pY = Y + dY_smooth*prediction_target
        pA = A + dA_smooth*prediction_target
        pX = pX[:-prediction_target]
        pY = pY[:-prediction_target]
        pA = pA[:-prediction_target]

        # Objective
        oX = X[prediction_target:]
        oY = Y[prediction_target:]
        oA = A[prediction_target:]

        # Error
        eX = oX - pX
        eY = oY - pY
        eA = oA - pA

        # SHUFFLE TEST
        dX_smooth = np.random.permutation(dX_smooth)
        dY_smooth = np.random.permutation(dY_smooth)
        dA_smooth = np.random.permutation(dA_smooth)

        # SHUFFLE TEST: Prediction
        pX = X + dX_smooth*prediction_target
        pY = Y + dY_smooth*prediction_target
        pA = A + dA_smooth*prediction_target
        pX = pX[:-prediction_target]
        pY = pY[:-prediction_target]
        pA = pA[:-prediction_target]

        # SHUFFLE TEST: Error
        eX_shuffle = oX - pX
        eY_shuffle = oY - pY
        eA_shuffle = oA - pA

        # Relative MAE (mean absolute error): Actual / Shuffle
        mae_X = np.mean(np.abs(eX))/np.mean(np.abs(eX_shuffle))
        mae_Y = np.mean(np.abs(eY))/np.mean(np.abs(eY_shuffle))
        mae_A = np.mean(np.abs(eA))/np.mean(np.abs(eA_shuffle))

        # Store
        mae_Xs.append(mae_X)
        mae_Ys.append(mae_Y)
        mae_As.append(mae_A)

    # Group Store
    group_mae_Xs.append(mae_Xs)
    group_mae_Ys.append(mae_Ys)
    group_mae_As.append(mae_As)

    # Report
    print("Group MAEs")
    print(np.mean(mae_Xs))
    print(np.mean(mae_Ys))
    print(np.mean(mae_As))

# Plot
plt.subplot(1,3,1)
plt.boxplot(group_mae_Xs)
plt.title('Relative X error')
plt.xlabel('controls - lesions')
plt.subplot(1,3,2)
plt.boxplot(group_mae_Ys)
plt.title('Relative Y error')
plt.xlabel('controls - lesions')
plt.subplot(1,3,3)
plt.boxplot(group_mae_As)
plt.title('Relative Angle error')
plt.xlabel('controls - lesions')
plt.show()

#FIN