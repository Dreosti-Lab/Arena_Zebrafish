DATAROOT = '/home/kampff/Data/Arena'
LIBROOT = 'C:/Users/thoma/OneDrive/Documents/GitHub/Arena_Zebrafish'

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
def analyze(tracking,path=False):
    
    if path:
        tracking = np.load(tracking)['tracking']
    
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
    plt.vlines(peaks, 0, 1200, 'r')
    plt.plot(smooth_bout_signal*20)
    plt.plot(fx)
    plt.plot(fy)
    plt.show()

    return bouts

# Label Bouts
def label(tracking, bouts):

    # Parameters
    FPS=120
    pre_window = 10
    post_window = 80
    num_frames = tracking.shape[0]
    num_bouts = bouts.shape[0]

    # Turn PC constant
    turn_pc = np.array( 
                        [4.45784725e-06,  7.29697833e-06,  8.34722354e-06,  7.25639602e-06,
                        6.83773435e-06,  1.05799488e-05,  9.59485594e-06,  1.04996460e-05,
                        9.50693646e-06,  6.68761575e-06,  1.74239537e-06, -5.13269107e-06,
                        -1.30955946e-05, -2.93123632e-05, -5.16772503e-05, -6.59745678e-05,
                        -6.24515957e-05, -6.82989320e-05, -5.84883171e-05, -5.49322933e-05,
                        -4.75273440e-05, -5.97750465e-05, -5.50942353e-05, -4.32771920e-05,
                        -4.53841833e-05, -4.39441043e-05, -4.29799500e-05, -3.66285781e-05,
                        -2.74927325e-05, -2.79482710e-05, -2.77149944e-05, -3.01089122e-05,
                        -2.69092862e-05, -2.75200069e-05, -3.25928317e-05, -3.87474743e-05,
                        -4.24973212e-05, -4.47429213e-05, -4.64712226e-05, -4.89719267e-05,
                        -5.91676326e-05, -6.22191781e-05, -6.21876092e-05, -6.47945016e-05,
                        -7.40367790e-05, -7.80097327e-05, -7.82331054e-05, -8.03180239e-05,
                        -8.55250976e-05, -8.88741024e-05, -8.93264800e-05, -9.13412355e-05,
                        -9.33324008e-05, -9.54639901e-05, -9.98497139e-05, -1.03221121e-04,
                        -1.08970275e-04, -1.13959552e-04, -1.20395095e-04, -1.22240153e-04,
                        -1.25032979e-04, -1.26145560e-04, -1.21958655e-04, -1.21565879e-04,
                        -1.21595218e-04, -1.18114363e-04, -1.17635286e-04, -1.12130918e-04,
                        -1.12562112e-04, -1.14707619e-04, -1.16066511e-04, -1.17252020e-04,
                        -1.22045156e-04, -1.22450517e-04, -1.25711027e-04, -1.25607020e-04,
                        -1.23958304e-04, -1.19578445e-04, -1.18268675e-04, -1.20917093e-04,
                        -1.23308934e-04, -1.18843590e-04, -1.19599994e-04, -1.20606743e-04,
                        -1.19085433e-04, -1.17407301e-04, -1.11223481e-04, -1.03411623e-04,
                        -9.72959419e-05, -9.09072743e-05, -3.92279029e-04, -8.75810372e-04,
                        -1.47534021e-03, -1.88185473e-03, -2.22179113e-03, -2.55991823e-03,
                        -2.84555972e-03, -3.18082206e-03, -3.41233583e-03, -3.70544285e-03,
                        -4.73103364e-03, -5.97680392e-03, -9.40038181e-03, -2.37417237e-02,
                        -5.71414180e-02, -7.90270203e-02, -8.59715002e-02, -8.39164195e-02,
                        -8.26775443e-02, -8.46991182e-02, -8.87082454e-02, -9.20826611e-02,
                        -9.44035333e-02, -9.58685766e-02, -9.77270940e-02, -9.94995655e-02,
                        -1.01423412e-01, -1.02874920e-01, -1.04038069e-01, -1.05218456e-01,
                        -1.06468904e-01, -1.07616346e-01, -1.08377944e-01, -1.09295619e-01,
                        -1.10020168e-01, -1.11017271e-01, -1.11630187e-01, -1.12289358e-01,
                        -1.13028781e-01, -1.13582258e-01, -1.14247743e-01, -1.14925706e-01,
                        -1.15475069e-01, -1.15872550e-01, -1.16510964e-01, -1.16891761e-01,
                        -1.17313917e-01, -1.17903131e-01, -1.18225351e-01, -1.18641475e-01,
                        -1.19053891e-01, -1.19258273e-01, -1.19559753e-01, -1.19870835e-01,
                        -1.20140247e-01, -1.20378214e-01, -1.20636915e-01, -1.20902923e-01,
                        -1.21193316e-01, -1.21443497e-01, -1.21709187e-01, -1.21760193e-01,
                        -1.21973109e-01, -1.22152281e-01, -1.22344918e-01, -1.22531978e-01,
                        -1.22724310e-01, -1.22906534e-01, -1.23223312e-01, -1.23339858e-01,
                        -1.23424650e-01, -1.23665608e-01, -1.23838407e-01, -1.24060679e-01,
                        -1.24108222e-01, -1.24361033e-01, -1.24545660e-01, -1.24807371e-01,
                        -1.25075108e-01, -1.25255340e-01, -1.25288654e-01, -1.25387074e-01,
                        -1.25516014e-01, -1.25501054e-01, -1.25552951e-01, -1.25657374e-01,
                        -1.25660401e-01, -1.25796678e-01, -1.25729603e-01, -1.25808149e-01]
                        )
   
    # Extract tracking
    X = tracking[:,2]
    Y = tracking[:,3]
    A = tracking[:,7]

    # Compute spatial and angular speed 
    speed_space, speed_angle=ARK_utilities.compute_bout_signals(X, Y, A)

    # Label bouts as turns (-1 = Left, 1 = Right) and swims (0)
    labels = np.zeros(num_bouts)
    for i, bout in enumerate(bouts):
        index = np.int(bout[0]) # Align to start
        if(index < pre_window):
            continue
        if(index > (num_frames-post_window)):
            continue
        tdD = speed_space[(index-pre_window):(index+post_window)]
        tD = np.cumsum(tdD)
        tdA = speed_angle[(index-pre_window):(index+post_window)]
        tA = np.cumsum(tdA)

        # Compare bout trajectory to Turn PC
        bout_trajectory = np.hstack((tD, tA))
        turn_score = np.sum(turn_pc * bout_trajectory)

        # Label
        if(turn_score < -90):
            labels[i] = -1
        if(turn_score > 90):
            labels[i] = 1

    return labels

#FIN