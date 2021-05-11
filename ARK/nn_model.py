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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

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
history_length = FPS*5 # 600
prediction_target = FPS*2 # 120

# Create Train/Test datasets
groups = [tracking_paths_controls, tracking_paths_lesions]
num_fish = len(tracking_paths_controls) + len(tracking_paths_lesions)

train_examples_per_fish = 2000
test_examples_per_fish = 500

num_train = train_examples_per_fish * num_fish
num_test = test_examples_per_fish * num_fish

train_set = np.zeros((num_train, history_length, 3))
test_set = np.zeros((num_test, history_length, 3))

train_goal = np.zeros((num_train, 3))
test_goal = np.zeros((num_test, 3))

train_counter = 0
test_counter = 0

for group in groups:
    for tracking_path in group:

        # Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
        tracking = np.load(tracking_path)['tracking']
        num_frames = tracking.shape[0]

        # Extract tracking
        X = tracking[:,2]
        Y = tracking[:,3]
        A = tracking[:,7]
        dA = ARK_utilities.diffAngle(A)
        dA = ARK_utilities.filterTrackingFlips(dA) / (2*np.pi)

        # Random train indices
        train_indices = np.random.randint(history_length, num_frames-prediction_target, train_examples_per_fish)
        for i in train_indices:
            history_range = np.arange(i - history_length, i)
            future_range = np.arange(i, i + prediction_target)

            A_history_cumulative = np.cumsum(dA[history_range])
            A_future_cumulative = np.cumsum(dA[future_range])

            train_set[train_counter,:,0] = X[history_range] - X[i]
            train_set[train_counter,:,1] = Y[history_range] - Y[i]
            train_set[train_counter,:,2] = A_history_cumulative - A_history_cumulative[-1]
            train_goal[train_counter, 0] = X[i + prediction_target] - X[i]
            train_goal[train_counter, 1] = Y[i + prediction_target] - Y[i]
            train_goal[train_counter, 2] = A_future_cumulative[-1]
            train_counter += 1

        # Random test indices
        test_indices = np.random.randint(history_length, num_frames-prediction_target, test_examples_per_fish)
        for i in test_indices:
            history_range = np.arange(i - history_length, i)
            future_range = np.arange(i, i + prediction_target)

            A_history_cumulative = np.cumsum(dA[history_range])
            A_future_cumulative = np.cumsum(dA[future_range])

            history_range = np.arange(i - history_length, i)
            test_set[test_counter,:,0] = X[history_range] - X[i]
            test_set[test_counter,:,1] = Y[history_range] - Y[i]
            test_set[test_counter,:,2] = A_history_cumulative - A_history_cumulative[-1]
            test_goal[test_counter, 0] = X[i + prediction_target] - X[i]
            test_goal[test_counter, 1] = Y[i + prediction_target] - Y[i]
            test_goal[test_counter, 2] = A_future_cumulative[-1]
            test_counter += 1

###############################################################################

# Define model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(history_length, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(3)
])

## Define model
#model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv1D(64, 3, activation='relu',input_shape=(history_length, 3)),
#  tf.keras.layers.Flatten(input_shape=(598, 3)),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dense(64, activation='relu'),
#  tf.keras.layers.Dense(32, activation='relu'),
#  tf.keras.layers.Dense(16, activation='relu'),
#  tf.keras.layers.Dense(3)
#])


# Specify loss function
loss_fn = tf.keras.losses.MeanAbsoluteError()

# Compile model (specify optimizer)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Fit
model.fit(train_set, train_goal, batch_size = 500, epochs=15)

# Evaluate
model.evaluate(test_set,  test_goal, verbose=2)

# Evaluate shuffle
suffled_indices = np.random.permutation(np.arange(num_test))
model.evaluate(test_set, test_goal[suffled_indices,:], verbose=2)

# Compute relative performance (actual / shuffle)
prediction = model.predict(test_set)
errors = test_goal - prediction 
errors_shuffled = test_goal[suffled_indices,:] - prediction
mae = np.mean(np.abs(errors), axis=0)
mae_shuffled = np.mean(np.abs(errors_shuffled), axis=0)
print(mae/mae_shuffled)


# Visualize
for idx in range (100):
    plt.plot(test_set[idx,:,0], test_set[idx,:,1])
    plt.plot(test_set[idx,-1,0], test_set[idx,-1,1], 'rx')
    plt.plot(test_goal[idx,0], test_goal[idx,1], 'ro')
    plt.plot(prediction[idx,0], prediction[idx,1], 'mo')
    plt.show()

#FIN