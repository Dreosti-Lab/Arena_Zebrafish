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
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
import ARK_utilities

# Reload libraries
import importlib
importlib.reload(ARK_utilities)

# Analyze Bouts
def analyze(tracking):

    # Extract tracking
    fx = tracking[:,0]
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]

    # Compute spatial and angular speed 
    speed_space, speed_angle=ARK_utilities.compute_bout_signals(bx, by, ort)
    
    # Absolute Value of angular speed
    speed_abs_angle = np.abs(speed_angle)

    # Detect negative/error values and set to zero
    bad_values = (area < 0) + (motion < 0) + (speed_space < 0)
    speed_space[bad_values] = 0.0
    speed_abs_angle[bad_values] = 0.0
    motion[bad_values] = 0.0

    # Weight contribution by STD
    std_space = np.std(speed_space)    
    std_angle = np.std(speed_abs_angle)    
    std_motion = np.std(motion)
    speed_space_norm = speed_space/std_space
    speed_angle_norm = speed_abs_angle/std_angle
    motion_norm = motion/std_motion

    # Sum weighted signals
    bout_signal = speed_space_norm + speed_angle_norm + motion_norm

    # Interpolate over bad values
    for i, bad_value in enumerate(bad_values):
        if bad_value == True:
            bout_signal[i] = bout_signal[i-1]

    # Smooth signal for bout detection   
    bout_filter = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    smooth_bout_signal = signal.fftconvolve(bout_signal, bout_filter, 'same')    

    # Determine Threshold levels
    # - Determine the largest 100 values and take the median
    # - Use 10% of max level, divide by 10, for the base threshold
    sorted_bout_signal = np.sort(smooth_bout_signal)
    max_norm = np.median(sorted_bout_signal[-100:])    
    upper_threshold = max_norm/10
    lower_threshold = upper_threshold/2

    # Find bouts (peaks)
    starts, peaks, stops = ARK_utilities.find_peaks_dual_threshold(smooth_bout_signal, upper_threshold, lower_threshold)
    numBouts = np.size(peaks)    
    bouts = np.zeros([numBouts, 8])

    # Set bout parameters
    for i in range(numBouts):

        start = starts[i] - 2   # Start frame (-2 frames)
        stop = stops[i]         # Stop frame

        x = bx[start:stop]      # X trajectory
        y = by[start:stop]      # Y trajectory

        eye_x = ex[start:stop]  # Eye X trajectory
        eye_y = ey[start:stop]  # Eye Y trajectory

        pre_x = bx[(start-20):start] # Preceding 20 frames X
        pre_y = by[(start-20):start] # Preceding 20 frames Y

        sx = x - x[0]   # Center X trajectory
        sy = y - y[0]   # Center Y trajectory

        # Get orientation prior to bout start (median of 5 preceding frames)
        align_ort = np.median(2*np.pi*(ort[(start-5):start] / 360.0))

        # Compute aligned distance (X = forward)
        ax = np.cos(align_ort) * sx - np.sin(align_ort) * sy
        ay = -1 * (np.sin(align_ort) * sx + np.cos(align_ort) * sy)

        # Create a heading vector (start)
        vx = np.cos(align_ort)
        vy = -1*np.sin(align_ort)

        # Create a heading vector (stop)
        final_ort = np.median(2*np.pi*(ort[stop:(stop+5)] / 360.0))
        vx = np.cos(final_ort)
        vy = -1*np.sin(final_ort)

        bouts[i, 0] = starts[i] - 2 # 2 frames before Upper threshold crossing 
        bouts[i, 1] = peaks[i]      # Peak frame
        bouts[i, 2] = stops[i]+1    # frame of Lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(speed_angle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sqrt(sx[-1]*sx[-1] + sy[-1]*sy[-1]) # Net distance change
        bouts[i, 6] = ax[-1]
        bouts[i, 7] = ay[-1]

    # Filter "tiny" bouts (net distance less than 1.5 pixels)
    not_tiny_bouts = bouts[:, 5] > 1.5
    bouts = bouts[not_tiny_bouts, :]

    # Debug
    #plt.vlines(peaks, 0, 1200, 'r')
    #plt.plot(smooth_bout_signal*20)
    #plt.plot(fx)
    #plt.plot(fy)
    #plt.show()

    return bouts

# Label Bouts
def label(tracking, bouts):

    # Extract tracking
    fx = tracking[:,0]
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]

    # Compute spatial and angular speed 
    speed_space, speed_angle=ARK_utilities.compute_bout_signals(bx, by, ort)

    # Label bouts as turns (l, r), routines (R, L), and swims (S), etc.
    initial_turns = []
    angles = bouts[:,4]
    distances = bouts[:,5]
    for bout in bouts:

        # Extract bout features
        start = int(bout[0])
        peak = int(bout[1])
        stop = int(bout[2])
        duration = int(bout[3])
        net_angle = bout[4]
        net_distance = bout[5]

        # Measure intial turn
        if(duration >= 10):
            initial_turns.append(np.sum(speed_angle[start:(start+10)]))
        else:
            initial_turns.append(0)
    initial_turns = np.array(initial_turns)

    # Classifly
    R = (initial_turns > 15) * (distances > 5)
    L = (initial_turns < -15) * (distances > 5)
    plt.plot(initial_turns, distances, '.')
    plt.show()

    return

#FIN