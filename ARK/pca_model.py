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

# Load PCA of bouts
pca_path = DATAROOT + '/pca.npz'
pca_mean = np.load(pca_path)['pca_mean']
pca_components = np.load(pca_path)['pca_components']
controls_compressed = np.load(pca_path)['controls_compressed']
lesions_compressed = np.load(pca_path)['lesions_compressed']

# Parameters
FPS=120
history_length = 10
prediction_target = 0
num_components = 20

# Model
groups = [controls_compressed, lesions_compressed]
group_MAEs_true = []
group_MAEs_random = []
for group in groups:
    num_bouts = group.shape[0]
    MAEs_true = []
    MAEs_random = []

    # Guess next bout based on bout history
    for index in range(num_bouts - history_length - prediction_target):

        # Average bout loadings
        avg_bout = np.mean(group[index:(index+history_length), :num_components], axis=0)
        
        # Compare with future bout
        next_bout = group[(index+history_length+prediction_target), :num_components]
        MAE_true = np.mean(np.abs(avg_bout - next_bout))
        
        # Compare with random bout
        random_index = np.random.randint(0, num_bouts)
        while(random_index == index):
            random_index = np.random.randint(0, num_bouts)
        random_bout = group[random_index, :num_components]
        MAE_random = np.mean(np.abs(avg_bout - random_bout))

        # Store
        MAEs_true.append(MAE_true)
        MAEs_random.append(MAE_random)

    # Group Store
    group_MAEs_true.append(np.array(MAEs_true))
    group_MAEs_random.append(np.array(MAEs_random))

    # Report
    print("Group MAEs")
    print(np.mean(MAEs_true))
    print(np.mean(MAEs_random))
    print(np.mean(np.array(MAEs_true)/np.mean(np.array(MAEs_random))))

# Plot
plt.subplot(1,3,1)
plt.boxplot(group_MAEs_true)
plt.title('True prediction error')
plt.xlabel('controls - lesions')
plt.subplot(1,3,2)
plt.boxplot(group_MAEs_random)
plt.title('Random prediction error')
plt.xlabel('controls - lesions')
plt.subplot(1,3,3)
plt.boxplot(np.array(group_MAEs_true)/np.array(group_MAEs_random))
plt.title('Relative prediction error')
plt.xlabel('controls - lesions')
plt.show()

#FIN