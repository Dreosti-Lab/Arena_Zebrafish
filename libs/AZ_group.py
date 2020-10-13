# -*- coding: utf-8 -*-
"""
 SZ_summary:
     - Social Zebrafish - Summary Analysis Functions

@author: thoma
"""
# -----------------------------------------------------------------------------
# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import AZ_analysis_testing as AZA
import AZ_utilities as AZU
import AZ_summary as AZS
#------------------------------------------------------------------------------
# Utilities for summarizing and plotting "Arena Zebrafish" experiments

#def showBoutThings(fx,fy,ort,n,FPS=120):
ret,_, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC, boutStarts, boutStops, motion_signal, dM = AZS.extractBouts(fx,fy, ort, name=[],savepath=[],FPS=120,plot=False, preWindow=100, postWindow=400,save=False)
plt.figure('allBoutsDist')
plt.plot(allBoutsDist[n])

[bx_mm],[by_mm] = AZU.convertToMm([fx],[fy]) 
boutDistsMax=AZA.findBoutMax(allBoutsDist[n])
boutDistsArea=AZA.findBoutArea(allBoutsDist[n])
boutDists=AZU.computeDistPerBout(bx_mm,by_mm,boutStarts[n],boutStops[n])

print('boutDists=' + str(boutDists))
print('boutDistMax=' + str(boutDistsMax))
print('boutDistArea=' + str(boutDistsArea))
print('boutAngle=' + str(boutAngles[n]))    
plt.figure('thisBoutDist')
thisBoutDist=allBoutsDist[n]
xFr=range(len(thisBoutDist))
x=np.divide(xFr,FPS)
plt.plot(x,thisBoutDist)
plt.ylabel('Velocity (mm/s)')
plt.xlabel('Time (s)')

plt.figure('thisBoutOrt')
thisBoutOrt=allBoutsOrt[n]
xFr=range(len(thisBoutOrt))
x=np.divide(xFr,FPS)
plt.plot(x,thisBoutOrt)
plt.ylabel('Relative bout heading (change)')
plt.xlabel('Time (s)')

plt.figure('thisBoutMotion')
thisBoutMotion=allBouts[n]
xFr=range(len(thisBoutMotion))
x=np.divide(xFr,FPS)
plt.plot(x,thisBoutMotion)
plt.ylabel('Motion Energy (AU)')
plt.xlabel('Time (s)')

plt.figure('thisBoutMotion')
thisBoutMotion=allBouts[n]
xFr=range(len(thisBoutMotion))
x=np.divide(xFr,FPS)
plt.plot(x,thisBoutMotion)
plt.ylabel('Motion Energy (AU)')
plt.xlabel('Time (s)')
###################################################################################################
def unpackDictFile(dicFileName):
## Unpacks an individual fish dictionary and returns the values
## Strict return procedure
    dic=np.load(dicFileName,allow_pickle=True).item()
    info=dic['info']
    name=info['Date']+'_'+info['Genotype']+'_'+info['Condition']+'_'+info['Chamber']+'_'+info['FishNo']
    data=dic['data']
    BPS=data['BPS']
    avgVelocity=data['avgVelocity']
    distPerFrame=data['distPerFrame']
    cumDist=data['cumDist']
    heatmap=data['heatmap']
    avgBoutAV=data['avgBout']['mean']
    avgBoutSD=data['avgBout']['SD']
    boutAmps=data['boutAmps']
    allBouts=data['allBouts']
    
    return dic,name,BPS,avgVelocity,distPerFrame,cumDist,heatmap,avgBoutAV,avgBoutSD,boutAmps,allBouts
###############################################################################
    
def extractBouts(fx,fy,dist, ort, FPS=120, startThreshold=0.04, stopThreshold=-0.04,plot=False, preWindow=50, postWindow=450):
## Compute activity level of the fish in bouts per second (BPS)
    preWindow=int(np.floor((preWindow/1000)*FPS))
    postWindow=int(np.floor((postWindow/1000)*FPS))
    motion_signal=ndimage.filters.gaussian_filter1d(AZU.motion_signal(fx,fy,ort),2)
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    dM=ndimage.filters.gaussian_filter1d(np.diff(motion_signal),2)

#    startThreshold=np.median(motion_signal)-(np.std(motion_signal[motion_signal<(np.median(motion_signal)*1.5)]))
    startThreshold = (np.std(dM[dM>(2*np.median(dM))])+(np.median(dM)))/2
    stopThreshold=startThreshold*-1
    print(str(startThreshold) + 'to start.' + str(stopThreshold) + 'to stop.')
    for i, m in enumerate(dM):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i+1)
                ii=i
        else:
            if m>stopThreshold and i>ii+50:
#            if  ((np.abs(np.mean(dM[i-3:i+3])))<np.std(motion_signal[(dM<startThreshold)-1]) and (np.mean(motion_signal[i-2:i+4])<np.median(motion_signal)*1.5) and (i>ii+postWindow)):
                moving = 0
                boutStops.append(i+1)
    # Extract all bouts (ignore last and or first, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
    if(boutStarts[0]<preWindow+1):
        boutStarts=boutStarts[1:] 
        boutStops=boutStops[1:]        

    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion_signal)/FPS  

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Extract the bouts; motion and orientation
    boutStarts = boutStarts[(boutStarts > preWindow) * (boutStarts < (len(motion_signal)-postWindow))]
    
    allBouts = np.zeros([len(boutStarts), (preWindow+postWindow)])
    allBoutsOrt = np.zeros([len(boutStarts), (preWindow+postWindow)])
    boutAngles = np.zeros(len(boutStarts))
    for b in range(0,len(boutStarts)):
#        allBouts[b,:] = dist[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)]; # extract velocity over this bout
        allBoutsOrt[b,:] = AZA.rotateOrt(ort[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)]); # extract heading for this bout (rotated to zero initial heading)
        boutAngles[b] = np.mean(allBoutsOrt[b,-2:-1])-np.mean(allBoutsOrt[b,0:1]) # take the heading before and after

    Lturns=np.zeros(len(boutAngles))    
    Rturns=np.zeros(len(boutAngles))    
    
    for i,angle in enumerate(boutAngles):
        if angle > 5: 
            Lturns[i]=1
        elif angle < -5:
            Rturns[i]=1
    Rturns=Rturns!=0
    Lturns=Lturns!=0
    LturnPC=np.sum(Lturns)/(np.sum(Lturns)+np.sum(Rturns))
    
    if plot:
        plt.figure('Bout finder Analysis')
        xFr=range(len(motion_signal))
        x=np.divide(xFr,FPS)
        plt.plot(x,motion_signal)
        boutStartMarker=np.zeros(len(motion_signal))
        boutMarker=np.zeros(len(motion_signal))
        for i in range(len(boutStarts)):
            boutStartMarker[boutStarts[i]]=10
            boutMarker[(boutStarts[i]-preWindow):(boutStarts[i]+postWindow)]=10
    
        plt.plot(x,boutStartMarker,'-r')
#        plt.plot(x,boutMarker)
        plt.fill_between(x,boutMarker,0,alpha=0.2,color='Orange')
        plt.title('Bout finder Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Motion energy (AU)')
        
    return boutsPerSecond, allBouts, allBoutsOrt, boutAngles, LturnPC, boutStarts, boutStops, motion_signal, dM
 
# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram

# Analyze bouts and pauses (individual stats)
def analyze_bouts_and_pauses(tracking, testROI, stimROI, visibleFrames, startThreshold, stopThreshold):
    
    # Extract tracking details
    bx = tracking[:,2]
    by = tracking[:,3]
    ort = tracking[:,7]
    motion = tracking[:,8]                
    
    # Compute normlaized arena coordinates
    nx, ny = SZA.normalized_arena_coords(bx, by, testROI, stimROI)
    
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Extract all bouts (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numBouts= len(boutStarts)
    bouts = np.zeros((numBouts, 10))
    for i in range(0, numBouts):
        bouts[i, 0] = boutStarts[i]
        bouts[i, 1] = nx[boutStarts[i]]
        bouts[i, 2] = ny[boutStarts[i]]
        bouts[i, 3] = ort[boutStarts[i]]
        bouts[i, 4] = boutStops[i]
        bouts[i, 5] = nx[boutStops[i]]
        bouts[i, 6] = ny[boutStops[i]]
        bouts[i, 7] = ort[boutStops[i]]
        bouts[i, 8] = boutStops[i] - boutStarts[i]
        bouts[i, 9] = visibleFrames[boutStarts[i]]
        
    # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numPauses = numBouts+1
    pauses = np.zeros((numPauses, 10))

    # -Include first and last as pauses (clipped in video)
    # First Pause
    pauses[0, 0] = 0
    pauses[0, 1] = nx[0]
    pauses[0, 2] = ny[0]
    pauses[0, 3] = ort[0]
    pauses[0, 4] = boutStarts[0]
    pauses[0, 5] = nx[boutStarts[0]]
    pauses[0, 6] = ny[boutStarts[0]]
    pauses[0, 7] = ort[boutStarts[0]]
    pauses[0, 8] = boutStarts[0]
    pauses[0, 9] = visibleFrames[0]
    # Other pauses
    for i in range(1, numBouts):
        pauses[i, 0] = boutStops[i-1]
        pauses[i, 1] = nx[boutStops[i-1]]
        pauses[i, 2] = ny[boutStops[i-1]]
        pauses[i, 3] = ort[boutStops[i-1]]
        pauses[i, 4] = boutStarts[i]
        pauses[i, 5] = nx[boutStarts[i]]
        pauses[i, 6] = ny[boutStarts[i]]
        pauses[i, 7] = ort[boutStarts[i]]
        pauses[i, 8] = boutStarts[i] - boutStops[i-1]
        pauses[i, 9] = visibleFrames[boutStops[i-1]]
    # Last Pause
    pauses[-1, 0] = boutStops[-1]
    pauses[-1, 1] = nx[boutStops[-1]]
    pauses[-1, 2] = ny[boutStops[-1]]
    pauses[-1, 3] = ort[boutStops[-1]]
    pauses[-1, 4] = len(motion)-1
    pauses[-1, 5] = nx[-1]
    pauses[-1, 6] = ny[-1]
    pauses[-1, 7] = ort[-1]
    pauses[-1, 8] = len(motion)-1-boutStops[-1]
    pauses[-1, 9] = visibleFrames[boutStops[-1]]
    return bouts, pauses
        
# Analyze temporal bouts
def analyze_temporal_bouts(bouts, binning):

    # Determine total bout counts
    num_bouts = bouts.shape[0]

    # Determine largest frame number in all bouts recordings (make multiple of 100)
    max_frame = np.int(np.max(bouts[:, 4]))
    max_frame = max_frame + (binning - (max_frame % binning))
    max_frame = 100 * 60 * 15 # 15 minutes

    # Temporal bouts
    visible_bout_hist = np.zeros(max_frame)
    non_visible_bout_hist = np.zeros(max_frame)
    frames_moving = 0
    visible_frames_moving = 0
    non_visible_frames_moving = 0
    for i in range(0, num_bouts):
        # Extract bout params
        start = np.int(bouts[i][0])
        stop = np.int(bouts[i][4])
        duration = np.int(bouts[i][8])
        visible = np.int(bouts[i][9])

        # Ignore bouts beyond 15 minutes
        if stop >= max_frame:
            continue

        # Accumulate bouts in histogram
        if visible == 1:
            visible_bout_hist[start:stop] = visible_bout_hist[start:stop] + 1
            visible_frames_moving += duration
        else:
            non_visible_bout_hist[start:stop] = non_visible_bout_hist[start:stop] + 1
            non_visible_frames_moving += duration
        frames_moving += duration

    #plt.figure()
    #plt.plot(visible_bout_hist, 'b')
    #plt.plot(non_visible_bout_hist, 'r')
    #plt.show()

    # Bin bout histograms
    visible_bout_hist_binned = np.sum(np.reshape(visible_bout_hist.T, (binning, -1), order='F'), 0)
    non_visible_bout_hist_binned = np.sum(np.reshape(non_visible_bout_hist.T, (binning, -1), order='F'), 0)

    #plt.figure()
    #plt.plot(visible_bout_hist_binned, 'b')
    #plt.plot(non_visible_bout_hist_binned, 'r')
    #plt.show()

    # Compute Ratio
    total_bout_hist_binned = visible_bout_hist_binned + non_visible_bout_hist_binned
    vis_vs_non = (visible_bout_hist_binned - non_visible_bout_hist_binned) / total_bout_hist_binned

    # Normalize bout histograms
    #visible_bout_hist_binned = visible_bout_hist_binned / frames_moving
    #non_visible_bout_hist_binned = non_visible_bout_hist_binned / frames_moving
    #vis_v_non = visible_bout_hist_binned / non_visible_bout_hist_binned

    # ----------------
    # Temporal Bouts Summary Plot
    #plt.figure()
    #plt.plot(vis_vs_non, 'k')
    #plt.ylabel('VPI')
    #plt.xlabel('minutes')
    #plt.show()

    return vis_vs_non

# FIN
