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

    # Filter negative/error values
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
    count = 0
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

        plt.plot(x, y, 'y', alpha=0.1)
        plt.plot(eye_x, eye_y, 'c', alpha=0.1)
        plt.plot(pre_x, pre_y, 'm', alpha=0.1)
        plt.plot(eye_x[0], eye_y[0], 'bo', alpha=0.1)
        plt.plot(x[0], y[0], 'go', alpha=0.1)

        # Create a heading vector (start)
        vx = np.cos(align_ort)
        vy = -1*np.sin(align_ort)
        plt.plot([x[0], x[0] + vx*5], [y[0], y[0] + vy*5], alpha=0.1)

        # Create a heading vector (stop)
        final_ort = np.median(2*np.pi*(ort[stop:(stop+5)] / 360.0))
        vx = np.cos(final_ort)
        vy = -1*np.sin(final_ort)
        plt.plot([x[-1], x[-1] + vx*5], [y[-1], y[-1] + vy*5], alpha=0.1)
        plt.plot(ax + x[0], ay + y[0], 'r', alpha=0.1)

        count = count + 1
        if(count == 1100):
            plt.show()

        bouts[i, 0] = starts[i] - 2 # 2 frames before Upper threshold crossing 
        bouts[i, 1] = peaks[i]      # Peak frame
        bouts[i, 2] = stops[i]+1    # frame of Lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(speed_angle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sqrt(sx[-1]*sx[-1] + sy[-1]*sy[-1]) # Net distance change
        bouts[i, 6] = ax[-1]
        bouts[i, 7] = ay[-1]


    # Debug
    #plt.vlines(peaks, 0, 1200, 'r')
    #plt.plot(smooth_bout_signal*20)
    #plt.plot(fx)
    #plt.plot(fy)
    #plt.show()

    return bouts

# Label Bouts
def label(bouts):


    # Label bouts as turns, swims, etc.
    for bout in bouts:
        print('hi')

    return

#FIN