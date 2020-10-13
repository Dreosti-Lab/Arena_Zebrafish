# -*- coding: utf-8 -*-
"""
Created on Wed May 07 19:13:12 2014

@author: Elena
"""
# -----------------------------------------------------------------------------
# Detect Platform
import platform
if(platform.system() == 'Linux'):
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
else:
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'

# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import AZ_utilities as AZU
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import AZ_math as AZM
import cv2 as cv2

## Dispersal analysis: take trajectory in 5 second windows and find the smallest circle that encompasses that trajectory. 
## Needs only fx, fy and framerate

def measureDispersal(xPos,yPos,frameRate=120,window=5):
    print('Measuring dispersal...')
    winFr=np.int(np.floor(window*frameRate))
    numFr=len(xPos)
    dispVec=np.zeros(numFr)
    
    for i in range(numFr): # cycle through all time points
#        print(i)
        if i>winFr: # skip first window
            if i==np.floor_divide(numFr,2): print('Halfway through dispersal measurement...')
            xPosj=[]
            yPosj=[]
            for j in range(i-winFr,i): # cycle through positions window secs behind
                # build this trajectory
                xPosj.append(xPos[j])
                yPosj.append(yPos[j])
            
            # find time points of most distant points in x and y
            x1_t=np.where(xPosj==np.min(xPosj))[0][0]
            x2_t=np.where(xPosj==np.max(xPosj))[0][0]
            y1_t=np.where(yPosj==np.min(yPosj))[0][0]
            y2_t=np.where(yPosj==np.max(yPosj))[0][0]
            
            # find their coordinates
            x1=([xPosj[x1_t],yPosj[x1_t]])
            x2=([xPosj[x2_t],yPosj[x2_t]])
            y1=([xPosj[y1_t],yPosj[y1_t]])
            y2=([xPosj[y2_t],yPosj[y2_t]])
            
            # find distance between each of these points and find largest
            xy=[]
            xy.append(AZU.computeDist(x1[0],x1[1],x2[0],x2[1]))
            xy.append(AZU.computeDist(x1[0],x1[1],y1[0],y1[1]))
            xy.append(AZU.computeDist(x1[0],x1[1],y2[0],y2[1]))
            xy.append(AZU.computeDist(x2[0],x2[1],y1[0],y1[1]))
            xy.append(AZU.computeDist(x2[0],x2[1],y2[0],y2[1]))
            xy.append(AZU.computeDist(y1[0],y1[1],y2[0],y2[1]))
            dispVec[i] = np.max(xy)/2 # take half the largest distance as radius for this time frame
    
    # pad the start of vector (first window seconds) with the first dispersal value 
    for i in range(winFr):
        dispVec[0:winFr]=dispVec[winFr+1]
        
    print('Finished measuring dispersal')
    return dispVec
#    # (OPTIONAL) Pick a part of the trajectory to display and draw a circle for presentations
#    if(plotCirc):
#        
#        CircMed_t=np.where(dispVec==np.median(dispVec))[0][0]
#        CircMax_t=np.where(dispVec==np.max(dispVec))[0][0]
#        
#        medRad=dispVec[CircMed_t]
#        maxRad=dispVec[CircMax_t]
#        
#        medx=xPos[CircMed_t]
#        medy=yPos[CircMed_t]
#        
#        maxx=xPos[CircMax_t]
#        maxy=yPos[CircMax_t]
#        
#        # grab trajectories
#        medxPosj=[]
#        medyPosj=[]
#        maxxPosj=[]
#        maxyPosj=[]
#        for j in range(CircMed_t-winFr,CircMed_t): # cycle through positions window secs behind
#            # build this trajectory
#            medxPosj.append(xPos[j])
#            medyPosj.append(yPos[j])
#        
#        for j in range(CircMax_t-winFr,CircMax_t): # cycle through positions window secs behind
#            # build this trajectory
#            maxxPosj.append(xPos[j])
#            maxyPosj.append(yPos[j])
#            
#        figName='MedTrajCirc'
#        plt.figure(figName)
#        plt.title(figName)
#        plt.plot(medxPosj,medyPosj)
#        H=(np.median(dispVec)*2)+5 # mm
#        lim=(np.sqrt(np.square(H)/2))
##        xst=np.min(medxPosj)-2.5
##        yst=np.min(medyPosj)-2.5
##        plt.xlim(xst,xst+lim)
##        plt.ylim(yst,yst+lim)
#        ax = plt.gca()
#        ylims=ax.get_ylim()
#        xlims=ax.get_xlim()
#        Xcen=np.mean(xlims)
#        Ycen=np.mean(ylims)
#        plt.xlim((Xcen-(lim/2)),Xcen+(lim/2))
#        plt.ylim((Ycen-(lim/2)),Ycen+(lim/2))
#        circlemed=plt.Circle((np.mean(xlims),np.mean(ylims)),medRad,color='blue',fill=False)
#        ax.add_artist(circlemed)
#        
#        figName='MaxTrajCirc'
#        plt.figure(figName)
#        plt.title(figName)
#        plt.plot(maxxPosj,maxyPosj)
#        H=(np.max(dispVec)*2)+5 # mm
#        lim=(np.sqrt(np.square(H)/2))
#        ax = plt.gca()
#        ylims=ax.get_ylim()
#        xlims=ax.get_xlim()
#        Xcen=np.mean(xlims)
#        Ycen=np.mean(ylims)
#        plt.xlim((Xcen-(lim/2)),Xcen+(lim/2))
#        plt.ylim((Ycen-(lim/2)),Ycen+(lim/2))
#        circlemax=plt.Circle((np.mean(xlims),np.mean(ylims)),maxRad,color='blue',fill=False)
#        ax.add_artist(circlemax)
    
   

def findBoutsXY(fx,fy,boutStarts):
    
    boutX = np.zeros(len(boutStarts))
    boutY = np.zeros(len(boutStarts))
    
    # cycle through boutStarts
    for i,s in enumerate(boutStarts):
        boutX[i]=fx[s]
        boutY[i]=fy[s]
#    print('Done')
    return boutX,boutY

def computeTimeCentreSurround(fx,fy,imageWidth=1280,imageHeight=1280,gridSize=40): # position of centre is hardcoded = only works for 20,20 heatmaps (default)
    
    ## this is the hard coded bit that means it only works with 40 x 40 gridSize at present
    startXH= 8
    startYH= 8
    endXH= 31
    endYH = 31
    
    locs=[]
    locs.append('Centre')
    locs.append('Surround')
    
    # create matrixs of zeros the same size as gridSize in hieght and width: boutCount, featureSum 
    startX= np.zeros(gridSize)
    startY= np.zeros(gridSize)
    endX= np.zeros(gridSize)
    endY = np.zeros(gridSize)
    
    for i in range(len(startX)): # cycle through i = startX
        div = imageWidth / gridSize # divide imageWidth by gridSize, 
        startX[i] = div * i 
        endX[i] = div * (i+1) 
        
        div = imageHeight / gridSize
        startY[i] = div * i
        endY[i] = div * (i+1) 
        
    pntCountCentre=0
    pntCountSurround=0
    for j in range(len(fx)): # cycle through all frames
        
        gridX = -1
        gridY = -1
        
        ## Is x position in centre?
        for k in range(len(locs)): # cycle through gridXLocations
            if fx[j] >startX[startXH] and fx[j]<endX[endXH]: # then we are in the central range of x
                gridX=k
                # Is y position in centre?
                for l in range(len(locs)): # cycle though gridYLocations
                    if fy[j] >startY[startYH] and fy[j] <endY[endYH]: # then we are in the central range of y
                        gridY=l
                        break
                
        if gridX>=0 and gridY>=0: # check we found this frame
            pntCountCentre += 1 # update count grid with location of this bout
        else: 
            pntCountSurround += 1
            
    pntCentreNorm=pntCountCentre/len(fx)
    pntSurroundNorm=pntCountSurround/len(fx)
    
    return pntCentreNorm,pntSurroundNorm
            
def heatmapFeature(fx,fy,boutStarts,feature,imageWidth=[],imageHeight=[],gridSize=20):
    
    # create matrixs of zeros the same size as gridSize in hieght and width: boutCount, featureSum 
    boutCount = np.zeros([gridSize,gridSize])
    featureSum = np.zeros([gridSize,gridSize])
    
    startX= np.zeros(gridSize)
    startY= np.zeros(gridSize)
    endX= np.zeros(gridSize)
    endY = np.zeros(gridSize)
    lostBoutsCount=0
    
    for i in range(len(startX)): # cycle through i = startX
        div = imageWidth / gridSize # divide imageWidth by gridSize, 
        startX[i] = div * i 
        endX[i] = div * (i+1) 
        
        div = imageHeight / gridSize
        startY[i] = div * i
        endY[i] = div * (i+1) 
    
    xBout,yBout = findBoutsXY(fx,fy,boutStarts)
    
    for j in range(len(xBout)): # cycle through bouts
        
        gridX = -1
        gridY = -1
        ## Find X position in grid for this bout
        for k in range(gridSize): # cycle through gridXLocations
            if xBout[j] >startX[k] and xBout[j]<endX[k]: # then we are in the right x
                gridX=k
                # Find y position
                for l in range(gridSize): # cycle though gridYLocations
                    if yBout[j] >startY[l] and yBout[j] <endY[l]: # then we are in the right y
                        gridY=l
                        break
                
        if gridX>=0 and gridY>=0: # check we found this bout
            boutCount[gridX,gridY] += 1 # update boutCount grid with location of this bout
            featureSum[gridX,gridY]=featureSum[gridX,gridY]+feature[j]
        else: 
            lostBoutsCount+=1
            
    ##  grids to PDF and means
    np.seterr(divide='ignore',invalid='ignore')
    boutDensity=np.divide(boutCount,np.sum(boutCount))
    featureDensity=np.divide(featureSum,np.sum(featureSum))
    featureAverage=np.divide(featureSum,boutCount)
    featureDensity[np.isnan(featureDensity)]=0
    featureAverage[np.isnan(featureAverage)]=0
    print('Lost ' + str(lostBoutsCount) + ' frames.')
    return boutCount,boutDensity,featureSum,featureDensity,featureAverage
    
def extractBouts(fx,fy, ort, distPerSec,name=[],savepath=[],FPS=120,plot=False, preWindow=100, postWindow=300,save=True):
    ret=1 # assume it works
    
    ## Compute activity level of the fish in bouts per second (BPS)
    preWindow=int(np.floor((preWindow/1000)*FPS))
    postWindow=int(np.floor((postWindow/1000)*FPS))
    _,motion_signal=AZU.motion_signal(fx,fy,ort)
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    dM=np.zeros(len(motion_signal))
    dM[1:]=ndimage.filters.gaussian_filter1d(np.diff(motion_signal),2)

    # thresholds should be computed for each movie. Plot out a histogram of the deltas in motion_signal. We assume the noise is Gaussian, so fit a Gaussian to the data. 5* the Sigma of this Gaussian should be a good threshold.
#    sortDM=sorted(np.abs(dM))
#    sampleDM=sortDM[0:int(np.floor(len(sortDM)*0.8))] # remove extreme 10% of values (i.e. ignore the ridiculously high motion signals as we want to estimate the noise)
    histDM,c=np.histogram(dM, bins=500) # plot it out in a histogram
    cc = (c[:-1]+c[1:])/2
    mp=np.max(histDM)
    sd=np.std(dM)
    initial_guess = [mp,0,sd]
    popt, pcov = curve_fit(AZM.gaussian_func, cc, histDM,p0=initial_guess)
    if plot:
        plt.scatter(cc,histDM)
        plt.plot(cc,AZM.gaussian_func(cc,*popt))
        plt.show() 
    
    # in order to be counted as a bout, the rate of increase must be at least 5 sigmas from the gaussian: appears robust when used on a slightly smoothed trace     
    # in order for the bout to end, it must fall to below 1 sigma for 5 consecutive frames and be at least 40 frames after the initiation of the bout (the latter probably not needed)
    startThreshold = popt[2]*4
    print('start thresh is ' + str(startThreshold))
    stopThreshold = popt[2]
    print('stop thresh is' + str(stopThreshold))
    smooth=AZU.smoothSignal(dM,5)
    for i, m in enumerate(smooth):
        br=0
        if(moving == 0):
            if br==1:
                break
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
                ii=i
        else:
            if np.abs(m)<stopThreshold and i>ii+40:
                for k in range(i,i+5):
                    if np.abs(dM[k]) > stopThreshold:
                        break
                    else:
                        moving = 0
    
    # Extract all bouts (ignore last and or first, if clipped)
    boutStarts = np.array(boutStarts)
    for i in range(0,len(boutStarts)):
        boutStops.append((boutStarts[i]-preWindow)+postWindow)
    boutStops = np.array(boutStops)
    
    while(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
    while(len(boutStops) > len(boutStarts)):
        boutStops = boutStops[:-1]
    if len(boutStarts)!=0:
        if(boutStarts[0]<preWindow+1):
            boutStarts=boutStarts[1:] 
            boutStops=boutStops[1:]        
    if len(boutStarts)!=0:
        if(boutStarts[len(boutStarts)-1]+postWindow>len(motion_signal)):
            boutStarts=boutStarts[:-1]
            boutStops=boutStops[:-1]
        
    # Return if failed to find any unclipped bouts
    if(len(boutStarts)==0 or len(boutStops)==0): 
        return -1,-1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1
    # threshold on maximum
#    keep = np.zeros(len(boutStarts))
#    for i,b in enumerate(boutStarts):
#        mm=np.max(AZU.smoothSignal(motion_signal[b-preWindow:b+postWindow],5))>1.2
#        if mm: keep[i]=1
#    keep=keep!=0
#    boutStarts=boutStarts[keep]
#    boutStops=boutStops[keep]
    
    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion_signal)/FPS  

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Extract the bouts; distance, motion_signal, orientation and angles
    boutStarts = boutStarts[(boutStarts > preWindow) * (boutStarts < (len(motion_signal)-postWindow))]

    allBoutsDist = np.zeros([len(boutStarts), (preWindow+postWindow)])    
    allBouts = np.zeros([len(boutStarts), (preWindow+postWindow)])
    allBoutsOrt = np.zeros([len(boutStarts), (preWindow+postWindow)])
    boutAngles = np.zeros(len(boutStarts))
    
    for b in range(0,len(boutStarts)):
        allBoutsDist[b,:] = distPerSec[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)] # extract velocity over this bout
        allBouts[b,:] = motion_signal[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)] # extract velocity over this bout
        allBoutsOrt[b,:] = rotateOrt(ort[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)]) # extract heading for this bout (rotated to zero initial heading)
        boutAngles[b] = np.mean(allBoutsOrt[b,-2:-1])-np.mean(allBoutsOrt[b,0:1]) # take the heading before and after

    Lturns=np.zeros(len(boutAngles))    
    Rturns=np.zeros(len(boutAngles))    
    FSwims=np.zeros(len(boutAngles))
    
    for i,angle in enumerate(boutAngles):
        if angle > 10: 
            Lturns[i]=1
        elif angle < -10:
            Rturns[i]=1
        else:
            FSwims[i]=1
    Rturns=Rturns!=0
    Lturns=Lturns!=0
    FSwims=FSwims!=0
    LturnPC=(np.sum(Lturns)/(np.sum(Lturns)+np.sum(Rturns)))*100
    
    if plot:
        plt.figure('Bout finder Analysis First min')
        xFr=range(len(motion_signal))
        x=np.divide(xFr,FPS)
        ef=FPS*1*60
        plt.plot(x[:ef],motion_signal[:ef])
        boutStartMarker=np.zeros(len(motion_signal[:ef]))
        boutMarker=np.zeros(len(motion_signal[:ef]))
        for i in range(len(boutStarts[boutStarts<(ef-postWindow)])):
            boutStartMarker[boutStarts[i]]=10
            boutMarker[(boutStarts[i]-preWindow):(boutStarts[i]+postWindow)]=10
    
        plt.plot(x[:ef],boutStartMarker,'-r',linewidth=0.5)
#        plt.plot(x,boutMarker)
        plt.fill_between(x[:ef],boutMarker,0,alpha=0.4,color='Orange')
        figName='Bout finder Analysis'
        plt.title(figName)
        plt.xlabel('Time (s)')
        plt.ylabel('Motion energy (AU)')
        
        if plot and save:
            savepath=savepath+ '\\BoutFinder\\'
            AZU.cycleMkDir(savepath)
            saveName=savepath + name+'_BoutFinder.png'
            plt.savefig(saveName,dpi=600)
            
    while(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
    while(len(boutStops) > len(boutStarts)):
        boutStops = boutStops[:-1]
        
    return ret,Rturns,Lturns,FSwims,boutsPerSecond, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC, boutStarts, boutStops, motion_signal, dM
  
# rotate the orientation trace so that the initial heading is zero
def rotateOrt(ort):
    ort_rot=np.zeros(len(ort))
    ort_init=ort[0]
    for i,thisOrt in enumerate(ort):
        o=thisOrt-ort_init
        oAbs=np.abs(o)
        if o>180:
            o=(180-(o-180))*-1
        elif o<-180:
            o=180-(oAbs-180)
            
        ort_rot[i]=o
    return ort_rot
        
        

## compute cumulative angle from heading
def computeCumulativeAngle(ort,plot=True,FPS=120):
    # rotate heading so starting heading is zero
    ortRot=rotateOrt(ort)
    
    ortDiff=np.diff(ortRot)# differentiate the headings
    ortDiff=AZU.filterTrackingFlips(ortDiff) # check for unlikely flips in the tracking caused by blurred eyes during movement (assumes turns of more than 100 degrees are impossible... which they are not)
    ortDiffAbs=np.abs(ortDiff) # take the absolute
    ort2pi=360-ortDiffAbs # subtract a circle from the absolute difference
    
    # we will now compare ort2pi and ortDiffAbs to accumulate the smallest in magnitude but keep the sign. 
    cumOrt=np.zeros(len(ortDiff)-1)
    for i in range(len(cumOrt)):
        if i !=0: 
           if ort2pi[i]<ortDiffAbs[i]:
               cumOrt[i]=ort2pi[i]+cumOrt[i-1]
           else:
               cumOrt[i]=ortDiff[i]+cumOrt[i-1]
    ortAbs=AZU.accumulate(ortDiffAbs)
    totAngle=ortAbs[-1]               
    cumAngle=cumOrt[-1]
    avAngVel=cumAngle/((len(cumOrt))/FPS)
    bias=cumAngle/totAngle # positive is left turns, negative is right turns

    if(plot):
        xFr=range(len(ort))
        x=np.divide(xFr,FPS)
        plt.figure()
        plt.plot(x,cumOrt)
        plt.title('Cumulative angle')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative angle (degrees from initial heading)')

    return avAngVel,bias,cumOrt


def unpackGroupDictFile(dicFileName):
    dic=np.load(dicFileName,allow_pickle=True).item()
    met=dic['Metrics']
    avg=dic['avgData']
    name=dic['Name']
    avgCumDistAV=met['cumDist']['Mean']
    avgCumDistSEM=met['cumDist']['SEM']
    avgBoutAV=met['avgBout']['Mean']
    avgBoutSEM=met['avgBout']['SEM']
    avgHeatmap=met['avgHeatmap']['Mean']
    avgVelocity=avg['avgVelocity']
    avgAngVelocityBouts=avg['avgAngVelocityBout']
    biasLeftBout=avg['biasLeftBout']
    LTurnPC=avg['LTurnPC']
    avgBoutAmps=avg['boutAmps']
    allBPS=avg['BPSs']
    
    return name,avgCumDistAV,avgCumDistSEM,avgBoutAmps,allBPS,avgBoutAV,avgBoutSEM,avgHeatmap,avgVelocity,avgAngVelocityBouts,biasLeftBout,LTurnPC    

def poolFishDict(dictList=[]):####NOT FINISHED OR REALLY NEEDED AT THIS POINT
    
    if(len(dictList)==0):
        print("Please provide a list of dictionaries to analyse (filepaths or dictionaries themselves)")
        return -1
    
    if(isinstance(dictList[0],str)):
        dictNameList=dictList
        dictList=[]
        for i,dicName in enumerate(dictNameList):
            dictList[i]=np.load(dicName,allow_pickle=True)
        
    elif(isinstance(dictList[0],dict)==False):
        print("Please provide a list of dictionaries to analyse (filepaths or dictionaries themselves)")
    
###############################################################################    
# Compute Viewing Preference Index
def computeVPI(xPositions, yPositions, testROI, stimROI, FPS=120):
     
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    visiblePositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "VISIBLE" in Y by comparing Y with the above calculated threshold
    socialTop = stimROI[1] < visiblePositionThreshold_Y
        
    # Check Y threshold 
    if socialTop:  
        AllVisibleFrames = yPositions < visiblePositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP   
    else:
        AllVisibleFrames = yPositions > visiblePositionThreshold_Y   # Opposite True/False array
    
    # Determine Non-Visible Fames
    AllNonVisibleFrames = np.logical_not(AllVisibleFrames)
   
    # Count Visible and Non-Visible Frames
    numVisibleFrames = np.float(np.sum(AllVisibleFrames))     # Final Sum of Visible Frames
    numNonVisibleFrames= np.float(np.sum(AllNonVisibleFrames))  # Final Sum of NON Visible Frames 
        
    # Compute VPI
    VPI = (numVisibleFrames-numNonVisibleFrames)/np.size(yPositions)

    # Determine number of frames in a five minute bin
    bin_size = 60 * FPS
    max_frame = bin_size * 15

    # Compute "binned" VPI
    if len(AllVisibleFrames) >= max_frame:
        visible_bins = np.sum(np.reshape(AllVisibleFrames[:max_frame].T, (bin_size, -1), order='F'), 0)
        non_visible_bins = np.sum(np.reshape(AllNonVisibleFrames[:max_frame].T, (bin_size, -1), order='F'), 0)
        VPI_bins = (visible_bins - non_visible_bins)/bin_size
    else:
        VPI_bins = np.empty(15) * np.nan

    return VPI, AllVisibleFrames, AllNonVisibleFrames, VPI_bins


# Compute Social Preference Index
def computeSPI(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
    socialPositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "social" in X and Y by comparing Tracking X and Y with the above calculated threshold
    socialTop = stimROI[1]<socialPositionThreshold_Y
    socialLeft = stimROI[0]<socialPositionThreshold_X
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # Check X threshold 
    if socialLeft:
        
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # True/False array
        
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    # Check Y threshold 
    if socialTop:
        
        AllSocialFrames_Y_TF= yPositions<socialPositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP
        
           
    else:
        AllSocialFrames_Y_TF= yPositions>socialPositionThreshold_Y   # Opposite True/False array
    
        
    AllSocialFrames_TF= np.logical_and(AllSocialFrames_X_TF,AllSocialFrames_Y_TF)  # Final SOCIAL True/False array
    AllNONSocialFrames_TF=np.logical_and(AllSocialFrames_X_TF, np.logical_not(AllSocialFrames_Y_TF))   # Final NON SOCIAL True/False array
    
    # Count Social and Non-Social Frames
    numSocialFrames = np.float(np.sum(AllSocialFrames_TF))     # Final Sum of Social Frames
    numNONSocialFrames= np.float(np.sum(AllNONSocialFrames_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
#        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_TF, AllNONSocialFrames_TF

# Compute normalized arena coordinates
def normalized_arena_coords(xPositions, yPositions, testROI, stimROI):
    
    # Rescale by chamber dimensions
    chamber_x_min = testROI[0]
    chamber_y_min = testROI[1]
    chamber_width_pixels = testROI[2]
    chamber_height_pixels = testROI[3]
    chamber_width_mm = 17
    chamber_height_mm = 42
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
    socialPositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "social" in X and Y by comparing Tracking X and Y with the above calculated threshold
    socialTop = stimROI[1]<socialPositionThreshold_Y
    socialLeft = stimROI[0]<socialPositionThreshold_X
        
    # Normalize positions to "social arena" coordinates (in mm): -X, -Y away from stim, +X, +Y towards stim
    norm_x = ((xPositions-chamber_x_min)/chamber_width_pixels) * chamber_width_mm
    norm_y = ((yPositions-chamber_y_min)/chamber_height_pixels) * chamber_height_mm
    
   # Place stim fish on right 
    if socialLeft:
        norm_x = (norm_x * -1) + chamber_width_mm
            
    # Place stim fish on top 
    if ~socialTop:
        norm_y = (norm_y * -1) + chamber_height_mm
        
    return norm_x, norm_y


def computeSPI_3fish(xPositions, yPositions, testROI, stimLeft):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
#   
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # If stimLeft is TRUE
    if stimLeft:
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # Social Left     
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    AllNONSocialFrames_X_TF=np.logical_not(AllSocialFrames_X_TF)   # Final NON SOCIAL True/False array

    # Count Social and Non-Social Frames
    numSocialFrames = np.float(np.sum(AllSocialFrames_X_TF))     # Final Sum of Social Frames
    numNONSocialFrames= np.float(np.sum(AllNONSocialFrames_X_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
#        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_X_TF, AllNONSocialFrames_X_TF


# Compute VISIBLE Frames (when the Test Fish could potentially see the Stim Fish)
def computeVISIBLE(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    visiblePositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "VISIBLE" in Y by comparing Y with the above calculated threshold
    socialTop = stimROI[1]<visiblePositionThreshold_Y
        
    # Check Y threshold 
    if socialTop:  
        AllVisibleFrames_Y_TF= yPositions<visiblePositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP   
    else:
        AllVisibleFrames_Y_TF= yPositions>visiblePositionThreshold_Y   # Opposite True/False array
    
    
    return AllVisibleFrames_Y_TF

def findBoutArea(allBouts, FPS=120):
    
    areaBouts=[]

    for i,bout in enumerate(allBouts):
#        print("I'm here on bout number " + str(i))
        plt.plot(bout)
        areaBouts.append(np.trapz(bout-np.min(bout), dx=1/FPS))
        
    return areaBouts
    
def findBoutMax(allBouts):
    
    ampBouts=np.zeros(len(allBouts))
    for i in range(len(allBouts)):
        ampBouts[i]=(np.max(allBouts[i]))
        
    return ampBouts

# Measure distance traveled during experiment (in mm)
def distance_traveled(bx, by, ROI):

    # Rescale by chamber dimensions
    chamber_width_pixels = ROI[2]
    chamber_height_pixels = ROI[3]
    chamber_width_mm = 17
    chamber_height_mm = 42
    
    # Sample position every 10 frames (10 Hz) and accumulate distance swum
    # - Only add increments greater than 0.5 mm
    num_frames = len(bx)
    prev_x = bx[0]
    prev_y = by[0]
    distance = 0
    for f in range(9,num_frames,10):
        dx = ((bx[f]-prev_x)/chamber_width_pixels) * chamber_width_mm
        dy = ((by[f]-prev_y)/chamber_height_pixels) * chamber_height_mm
        d = np.sqrt(dx*dx + dy*dy)
        if(d > 0.5):
            distance = distance + d
            prev_x = bx[f]
            prev_y = by[f]           
    
    return distance    

# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_SPI(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'/tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute SPI (NS)
            SPI, AllSocialFrames, AllNONSocialFrames = computeSPI(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return SPI, AllSocialFrames, AllNONSocialFrames
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_VISIBLE(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'/tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute AllVisibleFrames_Y_TFisible Frames
            AllVisibleFrames_Y_TF = computeVISIBLE(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return AllVisibleFrames_Y_TF
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_correlations(Test_folder, testNumber, Stim_folder, stimNumber, SocialFrames, corrLength, threshold):
    
    # Analyze Test
    trackingFile = Test_folder + r'/tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = Stim_folder + r'/tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames
    motion_test = motion_test[SocialFrames]
    motion_stim = motion_stim[SocialFrames]
    goodTracking = np.where((motion_test >= 0.0) * (motion_stim >= 0.0))
    goodTracking = goodTracking[0]
    motion_test = motion_test[goodTracking]
    motion_stim = motion_stim[goodTracking]
    
    # Threshold Motion Data to Remove Noise Correlations        
    baseline = np.median(motion_test)
    motion_test = motion_test - baseline
    sigma = np.std(motion_test)
    threshold_test = sigma*threshold
    motion_test = (motion_test > threshold_test) * motion_test

    baseline = np.median(motion_stim)
    motion_stim = motion_stim - baseline
    sigma = np.std(motion_stim)
    threshold_stim = sigma*threshold
    motion_stim = (motion_stim > threshold_stim) * motion_stim
    
    # Prepare "Motion" Arrays for Correlation Analysis (Pad with median "motion" value)
    padValue = np.median(motion_test)
    padding = np.zeros(corrLength)+padValue
    motion_test_padded = np.concatenate((padding, motion_test, padding), axis = 0)
    motion_stim_padded = np.concatenate((padding, motion_stim, padding), axis = 0)
    motion_test_padded_rev = motion_test_padded[::-1] # Scramble/Reverse Test
 
    # Compute Auto-Correlations
    auto_corr_test = np.correlate(motion_test_padded, motion_test, mode="valid")
    auto_corr_stim = np.correlate(motion_stim_padded, motion_stim, mode="valid")
    
    # Compute Cross-Correlations
    cross_corr = np.correlate(motion_test_padded, motion_stim, mode="valid")
    cross_corr_rev = np.correlate(motion_test_padded_rev, motion_stim, mode="valid")
    
    # Make Correlation Data Structure (2D array)
    corr_data = np.vstack((auto_corr_test, auto_corr_stim, cross_corr, cross_corr_rev))
    
    return corr_data

# Analyze Bouts of Test and Stimulus Fish
def analyze_bouts(testFolder, testNumber, stimFolder, stimNumber, visibleFrames, btaLength, threshold, testROI, stimROI):
    
    # Analyze Test
    trackingFile = testFolder + r'/tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_test = tracking[:,0] 
    fy_test = tracking[:,1]
    bx_test = tracking[:,2]
    by_test = tracking[:,3]
    ex_test = tracking[:,4]
    ey_test = tracking[:,5]
    area_test = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = stimFolder + r'/tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_stim = tracking[:,0] 
    fy_stim = tracking[:,1]
    bx_stim = tracking[:,2]
    by_stim = tracking[:,3]
    ex_stim = tracking[:,4]
    ey_stim = tracking[:,5]
    area_stim = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames (set to 0) - interpolate?
    motion_test[motion_test < 0.0] = 0.0
    motion_stim[motion_stim < 0.0] = 0.0
    
    # Compute Signal for bout detection (smoothed motion signal)   
    bout_filter = np.array([0.25, 0.25, 0.25, 0.25])
    boutSignal_test = signal.fftconvolve(motion_test, bout_filter, 'same')    
    boutSignal_stim = signal.fftconvolve(motion_stim, bout_filter, 'same')

    # Determine Threshold levels
    # - Determine the largest 100 values and take the median
    # - Use 10% of max level, divide by 10, for the base threshold (sigma)
    sorted_motion = np.sort(motion_test)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_test = max_norm/10
    threshold_test = sigma_test*threshold    
    # - - - -
    print (threshold_test, max_norm)
    # - - - -
    sorted_motion = np.sort(motion_stim)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_stim = max_norm/10
    threshold_stim = sigma_stim*threshold
    
    # Extract Bouts from Tracking Data
    bouts_test = SZU.extract_bouts_from_motion(bx_test, by_test, ort_test, boutSignal_test, threshold_test, threshold_test-sigma_test, testROI, True)
    bouts_stim = SZU.extract_bouts_from_motion(bx_stim, by_stim, ort_stim, boutSignal_stim, threshold_stim, threshold_stim-sigma_stim, stimROI, False)
    
    # Get Info about each bout (Align on STARTS!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)
    
    # Position at bout onset, offset

    # On Social(Visible side) at bout onset
    visible_testDuringStim = visibleFrames[peaks_stim]
    visible_stimDuringTest = visibleFrames[peaks_test]

    # Orientation of other fish during bout Peak
    # Zero degrees is towards bout generating fish, 180 is away, 90 is facing with REye, -90 is LEye
    # Seperate Orientations based on Chamber (1-6)
    # 1 - test right
    # 2 - test left
    # 3 - test right
    # 4 - test left
    # 5 - test right
    # 6 - test left

    ortOfTestDuringStim = ort_test[peaks_stim]
    ortOfStimDuringTest = ort_stim[peaks_test]
    
    # Adjust orientations so 0 is always pointing towards "other" fish
    if testNumber%2 == 0: # Test Fish facing Left
        for i,ort in enumerate(ortOfTestDuringStim):
            if ort >= 0: 
                ortOfTestDuringStim[i] = ort - 180
            else:
                ortOfTestDuringStim[i] = ort + 180    
    if stimNumber%2 == 1: # Stim fish facing Left
        for i,ort in enumerate(ortOfStimDuringTest):
            if ort >= 0: 
                ortOfStimDuringTest[i] = ort - 180
            else:
                ortOfStimDuringTest[i] = ort + 180  

    # Concatenate into Bouts Structure    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(visible_stimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(visible_testDuringStim))))    
    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(ortOfStimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(ortOfTestDuringStim))))

    return bouts_test, bouts_stim

# Compute BTA of Test and Stimulus Fish for different measures
def compute_BTA(bouts_test, bouts_stim, output_test, output_stim, btaLength):
        
    # Get Info about each bout (Align on Peaks!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)

    # Allocate Space for BTAs    
    BTA_test = np.zeros((np.size(peaks_test),2*btaLength,2))
    BTA_stim = np.zeros((np.size(peaks_stim),2*btaLength,2))

    # Pad OUTPUT variable for alignment
    padValue = np.median(output_test)
    padding = np.zeros(btaLength)+padValue
    output_test_padded = np.concatenate((padding, output_test, padding), axis = 0)
    padValue = np.median(output_stim)
    padding = np.zeros(btaLength)+padValue
    output_stim_padded = np.concatenate((padding, output_stim, padding), axis = 0)

    # Compute Burst Triggered Average (Auto-Corr)
    BTA_test[:,:,0] = SZU.burst_triggered_alignment(peaks_test, output_test_padded, 0, btaLength*2)
    BTA_stim[:,:,0] = SZU.burst_triggered_alignment(peaks_stim, output_stim_padded, 0, btaLength*2)

    # Compute Burst Triggered Average
    BTA_test[:,:,1] = SZU.burst_triggered_alignment(peaks_test, output_stim_padded, 0, btaLength*2)
    BTA_stim[:,:,1] = SZU.burst_triggered_alignment(peaks_stim, output_test_padded, 0, btaLength*2)
    
    return BTA_test, BTA_stim
    
# FIN
    