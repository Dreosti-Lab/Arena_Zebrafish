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
import glob

# Peak Detection
def find_peaks_dual_threshold(values, upper_threshold, lower_threshold):    
    over = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[5:-100]:
        if over == 0:
            if values[i] > upper_threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i)
        else: #This is what happens when over the upper_threshold
            if values[i] > curPeakVal:
                curPeakVal = values[i]
                curPeakIdx = i
            elif values[i] < lower_threshold:
                over = 0
                curPeakVal = 0
                peaks.append(curPeakIdx)
                stops.append(i)
    
    return starts, peaks, stops

def diffAngle(Ort):
## Computes the change in angle over all frames of given Ort tracking data
    dAngle = np.diff(Ort)
    new_dAngle = [0]    
    for a in dAngle:
        if a < -270:
            new_dAngle.append(a + 360)
        elif a > 270:
            new_dAngle.append(a - 360)
        else:
            new_dAngle.append(a)
    
    return np.array(new_dAngle)

def filterTrackingFlips(dAngle):
## Identifies and reverses sudden flips in orientation caused by errors in tracking the eyes vs the body resulting in very high frequency tracking flips    
    new_dAngle = []    
    for a in dAngle:
        if a < -100:
            new_dAngle.append(a + 180)
        elif a > 100:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)

def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed

def motion_signal(X, Y, Ort):
    # Combines velocity and angular velocity each weighted by their standard deviation to give a combined 'motion_signal' metric of movement. 
    SpeedXY, SpeedAngle=compute_bout_signals(X, Y, Ort)
    
    # Absolute Value of angular speed
    SpeedAngle = np.abs(SpeedAngle)

    # Weight contribution by STD
    std_XY = np.std(SpeedXY)    
    std_Angle = np.std(SpeedAngle)    
    SpeedXY = SpeedXY/std_XY
    SpeedAngle = SpeedAngle/std_Angle

    # Sum Combined Signal
    motion_signal = SpeedXY+SpeedAngle

    return SpeedXY,SpeedAngle,motion_signal

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals(X, Y, Ort):

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
    # Filter Speed for outliers
    sigma = np.std(speedXY)
    baseline = np.median(speedXY)
    speedXY[speedXY > baseline+10*sigma] = -1.0
    
    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
        
    return speedXY, speedAngle

# Extract Bouts from Motion Signal
def extract_bouts_from_motion(X, Y, Ort, motion, upper_threshold, lower_threshold, ROI, test):

    if test:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, True)
    else:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, False)        
     
    # Find Peaks in Motion Signal 
    starts, peaks, stops = find_peaks_dual_threshold(motion, upper_threshold, lower_threshold)
    numBouts = np.size(peaks)    
    bouts = np.zeros([numBouts, 6])

    for i in range(numBouts):
        bouts[i, 0] = starts[i]-4 # Point 4 frames (40 ms) before Upper threshold crossing 
        bouts[i, 1] = peaks[i] # Peak
        bouts[i, 2] = stops[i]+1 # Point 1 frame (10 ms) after lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(SpeedAngle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sum(SpeedXY[starts[i]:stops[i]]) # Net distance change

    return bouts

def polar_orientation(Ort):
## Generates a polar histogram of the orientation of the fish
    ort_hist, edges = np.histogram(Ort, 18, (0, 360))
    plt.plot(edges/(360/(2*np.pi)), np.append(ort_hist, ort_hist[0]))
    max_ort = edges[np.argmax(ort_hist)]
    return max_ort

#filepath='D:\\\\StimFiles\\\\210325_EmxGFP_Ctrl_B2_1.csv'
#loomStarts,loomEnds,respEnds,loomPosX,loomPosY=findLoomsFromFile(filepath)
# FIN