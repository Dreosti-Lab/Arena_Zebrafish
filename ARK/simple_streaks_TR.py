LIBROOT = 'C:\Users\Tom\Documents\GitHub\Arena_Zebrafish'

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
# tracking_paths_controls = []
# tracking_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/EC_M0/*.npz")
#tracking_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*.npz")

# Get tracking files (ablation)
# tracking_paths_lesions = []
# tracking_paths_lesions += glob.glob(DATAROOT + "/GroupedTracking/EA_M0/*.npz")
#tracking_paths_lesions += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*.npz")
folderListFile_Ctrl='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Sham.txt'
folderListFile_Cond='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Lesion.txt'
# Parameters
FPS=120
def getFiles(folderNames):
    trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles=[],[],[],[],[],[]
    for folder in folderNames:
        trackingFolder=folder + r'\\Tracking\\'
        # Grab tracking files from folder or .txt folder list file
        trackingFilest=glob.glob(trackingFolder+'*tracking.npz')
        tailXFilest=glob.glob(trackingFolder+'*SegX.csv')
        tailYFilest=glob.glob(trackingFolder+'*SegY.csv')
        boutFilest=glob.glob(trackingFolder+'Analysis\*bouts*')
        tailFilest=glob.glob(trackingFolder+'Analysis\*tailAnalysis*')
        bodyThetaFilest=glob.glob(trackingFolder+'Analysis\*bodyTheta*')
        for i,s in enumerate(trackingFilest): #
            trackingFiles.append(s)
            tailXFiles.append(tailXFilest[i])
            tailYFiles.append(tailYFilest[i])
            boutFiles.append(boutFilest[i])
            tailFiles.append(tailFilest[i])
            bodyThetaFiles.append(bodyThetaFilest[i])
    return trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles

_,folderNames = AZU.read_folder_list(folderListFile_Ctrl)
_,folderNames1 = AZU.read_folder_list(folderListFile_Cond)
trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles=getFiles(folderNames)
trackingFiles1,tailXFiles1,tailYFiles1,tailFiles1,boutFiles1,bodyThetaFiles1=getFiles(folderNames1)

# Label bouts and measure streaks
groups = [tracking_paths_controls, tracking_paths_lesions]
group_streaks_true = []
group_streaks_random = []
for group in groups:
    streaks_true = []
    streaks_random = []
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Analyze bouts
        bouts = ARK_bouts.analyze(tracking)
        num_bouts = bouts.shape[0]

        # Label bouts
        labels_true = ARK_bouts.label(tracking, bouts)
        print(np.mean(labels_true))
        print(np.mean(np.abs(labels_true)))

        # Randomize labels
        labels_random = np.random.permutation(labels_true)

        # Measure streaks (true)
        prev_label = labels_true[0]
        current_streak = 1
        for label in labels_true[1:]:
            # Ignore swims?
            if label == 0:
                continue
            if label == prev_label:
                current_streak = current_streak + 1
            else:
                streaks_true.append(current_streak)
                prev_label = label
                current_streak = 1

        # Measure streaks (random)
        prev_label = labels_random[0]
        current_streak = 1
        for label in labels_random[1:]:
            # Ignore swims?
            if label == 0:
                continue
            if label == prev_label:
                current_streak = current_streak + 1
            else:
                streaks_random.append(current_streak)
                prev_label = label
                current_streak = 1

        # Report progress
        print('DONE: ' + tracking_path)

    # Store each group's streaks
    group_streaks_true.append(streaks_true)
    group_streaks_random.append(streaks_random)

# Report results
print("Controls: {0} vs {1}".format(np.mean(group_streaks_true[0]), np.mean(group_streaks_random[0])))
print("Lesions: {0} vs {1}".format(np.mean(group_streaks_true[1]), np.mean(group_streaks_random[1])))

all_streaks_true_controls = np.array(group_streaks_true[0])
all_streaks_random_controls = np.array(group_streaks_random[0])
all_streaks_true_lesions = np.array(group_streaks_true[1])
all_streaks_random_lesions = np.array(group_streaks_random[1])

hist_streaks_true_controls = np.cumsum(np.histogram(all_streaks_true_controls, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_random_controls = np.cumsum(np.histogram(all_streaks_random_controls, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_true_lesions = np.cumsum(np.histogram(all_streaks_true_lesions, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_random_lesions = np.cumsum(np.histogram(all_streaks_random_lesions, bins = np.arange(-0.5, 16, 1), density=1)[0])

plt.plot(hist_streaks_true_controls, 'b')
plt.plot(hist_streaks_random_controls, 'b--')
plt.plot(hist_streaks_true_lesions, 'r')
plt.plot(hist_streaks_random_lesions, 'r--')
plt.show()

long_streaks_true_controls = np.sum(all_streaks_true_controls > 10) / len(all_streaks_true_controls)
long_streaks_random_controls = np.sum(all_streaks_random_controls > 10) / len(all_streaks_random_controls)
long_streaks_true_lesions = np.sum(all_streaks_true_lesions > 10) / len(all_streaks_true_lesions)
long_streaks_random_lesions = np.sum(all_streaks_random_lesions > 10) / len(all_streaks_random_lesions)

# Report results
print("Controls: {0} vs {1}".format(long_streaks_true_controls, long_streaks_random_controls))
print("Lesions: {0} vs {1}".format(long_streaks_true_lesions, long_streaks_random_lesions))

#FIN