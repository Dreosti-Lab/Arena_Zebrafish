# -*- coding: utf-8 -*-
"""
 SZ_summary:
     - Social Zebrafish - Summary Analysis Functions

@author: adamk
"""
# -----------------------------------------------------------------------------
# Set Library Paths
lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Import useful libraries
import numpy as np
#------------------------------------------------------------------------------
# Utilities for summarizing and plotting "Arena Zebrafish" experiments
def populateSingleDictionary(date='',
                             gType='',
                             cond='',
                             chamber='',
                             fishNo=-1,
                             trackingFile='',
                             aviFile='',
                             BPS=-1,
                             avgVelocity=-1,
                             distPerFrame=-1,
                             cumDist=-1,
                             avgBout=-1,
                             avgBoutSD=-1,
                             boutAmps=-1,
                             boutDists=-1,
                             boutAngles=-1,
                             heatmap=-1,
                             cumOrt=-1,
                             avgAngVelocityBout=-1,
                             bias=-1,
                             boutSeq=-1,
                             allBoutsList=[],
                             allBoutsOrtList=[],
                             allBoutsDistList=[],
                             comb1=-1,
                             comb2=-1,
                             params=[]):
    
    if(len(params)!=0):
        date                =   params[0]
        gType               =   params[1]
        cond                =   params[2]
        chamber             =   params[3]
        fishNo              =   params[4]
        trackingFile        =   params[5]
        aviFile             =   params[6]
        BPS                 =   params[7]
        avgVelocity         =   params[8]
        distPerFrame        =   params[9]
        cumDist             =   params[10]
        avgBout             =   params[11]
        avgBoutSD           =   params[12]
        boutAmps            =   params[13]
        boutDists           =   params[14]
        boutAngles          =   params[15]
        heatmap             =   params[16]
        avgAngVelocityBout  =   params[17] 
        bias                =   params[18]
        LturnPC             =   params[19]     
        boutSeq             =   params[20]
        seqProbs1           =   params[21]
        seqProbs2_V         =   params[22]
        seqProbs2_P         =   params[23]
        seqProbs2_Z         =   params[24]
        boutStarts          =   params[25]
    
    SingleFish  =       {'info' :   {'Date'                 :   date,
                                     'Genotype'             :   gType,
                                     'Chamber'              :   chamber,
                                     'Condition'            :   cond,
                                     'FishNo'               :   fishNo,
                                     'TrackingPath'         :   trackingFile,
                                     'AviPath'              :   aviFile
                                     },
                         'data' :   {'BPS'                  :   BPS,
                                     'boutStarts'           :   boutStarts,
                                     'avgVelocity'          :   avgVelocity,
                                     'avgAngVelocityBout'   :   avgAngVelocityBout,
                                     'biasLeftBout'         :   bias,
                                     'LTurnPC'              :   LturnPC,
                                     'distPerFrame'         :   distPerFrame,
                                     'cumDist'              :   cumDist,
                                     'heatmap'              :   heatmap,
                                     'avgBout'              :   {'Mean'             :   avgBout,
                                                                 'SD'               :   avgBoutSD
                                                                 },
                                     'boutAmps'             :   boutAmps,
                                     'boutDists'            :   boutDists,
                                     'boutAngles'           :   boutAngles,
                                     'boutSeq'              :   boutSeq,
                                     'seqProbs1'            :   {'comb'     :   comb1,
                                                                 'prob'     :   seqProbs1},
                                     'seqProbs2'            :   {'comb'     :   comb2,
                                                                 'prob'     :   seqProbs2_V,
                                                                 'pvalues'  :   seqProbs2_P,
                                                                 'zscores'  :   seqProbs2_Z},
                                     'allBouts'             :   allBoutsList,
                                     'allBoutsDist'         :   allBoutsDistList,
                                     'allBoutsOrts'         :   allBoutsOrtList,
                                     }
                        }
    
    return SingleFish







###############################################################################
def populateGroupDictionary(groupName='default',groupDescriptParams=[],groupPooledData=[],allFish=[]):
## Populates a group dictionary with summary statistics, pooled data sets for convenient plotting, and links to individual fish in the group
## Strict input procedure
    
    GroupedFish    =    {'Name'                     :   groupName,
                         'Ind_fish'                 :   allFish,
                         'PooledData'               :   {'avgVelocities'      :   groupPooledData[0],
                                                         'cumDists'           :   groupPooledData[1],
                                                         'avgHeatmaps'        :   groupPooledData[2],
                                                         'avgBouts'           :   groupPooledData[3],
                                                         'boutAmps'           :   groupPooledData[4],
                                                         'BPSs'               :   groupPooledData[5]},
                         'Metrics'                  :   {'avgVelocity'        :     {'Mean'    :   groupDescriptParams[0],
                                                                                     'SEM'     :   groupDescriptParams[1]},
                                                         'cumDist'            :     {'Mean'    :   groupDescriptParams[2],
                                                                                     'SEM'     :   groupDescriptParams[3]},
                                                         'avgHeatmap'         :     {'Mean'    :   groupDescriptParams[4],
                                                                                     'SEM'     :   groupDescriptParams[5]},
                                                         'avgBout'            :     {'Mean'    :   groupDescriptParams[6],
                                                                                     'SEM'     :   groupDescriptParams[7]},
                                                         'boutAmp'            :     {'Mean'    :   groupDescriptParams[8],
                                                                                     'SEM'     :   groupDescriptParams[9]},
                                                         'BPS'                :     {'Mean'    :   groupDescriptParams[10],
                                                                                     'SEM'     :   groupDescriptParams[11]}},
                         }

    return GroupedFish
###############################################################################
    
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
    

# Define a gaussian function with offset
def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))
 
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
