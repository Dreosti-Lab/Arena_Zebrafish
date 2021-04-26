# -*- coding: utf-8 -*-
"""
Created on Mon Nov 04 13:58:42 2019

@author: Tom Ryan (Dreosti Lab, UCL)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Arena Zebrafish Repo
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------

import os
# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
# Import local modules
import AZ_figures as AZF
import AZ_utilities as AZU
import AZ_analysis as AZA
import AZ_summary as AZS
import AZ_streakProb as AZP

folderListFile=[]
#trackingFolder=[]
#Drive=r'D:\\'
folder=r'D:\\Movies\\RawMovies\\aspB2\\120Hz'
TrackingFolder=folder + r'\\Tracking\\'
DictionaryFolder=folder + r'\\Analysis\\Dictionaries\\'
figureFolder=folder + r'\\Analysis\\Figures\\'
AZU.cycleMkDir(DictionaryFolder)
AZU.cycleMkDir(figureFolder)
# Set FPS of camera
FPS = 120
# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile =r'F:\Data\Movies\Loom+Para\Loom+Para.txt'

# OR

#trackingFolder = r'D:\\Movies\\GroupedData\\Groups\\para\\'
#suff='EmxGFP_Asp_B0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='EmxGFP_Ctrl_B0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='WT_B0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='EmxGFP_Asp_M0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='para' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='WT_R0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
#suff='WT_M0_201115'
dateSuff=(datetime.date.today()).strftime("%y%m%d")
#suff=''
sf=0*60*FPS
ef=-1

# Set Flags
overwrite=True
createDict=True
createFigures=True
keepFigures=False
group=False # if set to TRUE change groupName above (ln 38)
createGroupFigures=True
keepGroupFigures=False
omitForward=False

# cycle through the start and end frames we want
#sfL=[0,36000,72000,108000,144000,180000,216000,252000,288000,324000]
#efL=[36000,72000,108000,144000,180000,216000,252000,288000,324000,360000]
#for timeLoop in range(len(efL)):
#    sf=sfL[timeLoop]
#sf=0
#ef=30*60*120


#    ef=efL[timeLoop]
#    pstr=' ' + str((sf/120)/60) + '-' + str((ef/120)/60) + 'mins'

trackingFiles=[]
dictList=[]
dictNameList=[]
#    groupName=nn + pstr

# Grab tracking files from folder or .txt folder list file
#trackingFiles=AZU.getTrackingFilesFromFolder(suff=suff,folderListFile=folderListFile,trackingFolder=trackingFolder)
# Or just search the TrackingData Directory
trackingFiles=glob.glob(TrackingFolder+'*.npz')
missingFiles=[]

# check through to see if analysis has already been done
print('Checking existing analysis files...')
dictOutNames=[]
for i,trackingFile in enumerate(trackingFiles):
    d,spl=trackingFile.rsplit(sep='\\',maxsplit=1)
    spl=spl[0:-13]
    dictFile=DictionaryFolder+spl+'_ANALYSIS.npy'
    if os.path.exists(dictFile):
        if overwrite==False:
            print(trackingFile + ' has already been analysed... removing from list')
            trackingFiles.remove(trackingFile)
            missingFiles.append(trackingFile)
        else:
            print(trackingFile + ' has already been analysed... overwriting...')
            dictOutNames.append(dictFile)
    else:
        dictOutNames.append(dictFile)
# run through each experiment
for k,trackingFile in enumerate(trackingFiles):
    print('Analysing ' + trackingFile)
    wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
    fx,fy,bx,by,ex,ey,area,ort,_=AZU.grabTrackingFromFile(trackingFile)
    if(ef>len(fx)) : ef = -1
    fx=fx[sf:ef]
    fy=fy[sf:ef]
    bx=bx[sf:ef]
    by=by[sf:ef]
    ex=ex[sf:ef]
    ey=ey[sf:ef]
    area=area[sf:ef]
    ort=ort[sf:ef]
    # How long is this movie?
    numFrames=fx.shape[0]
    numSecs=(numFrames/FPS)
    xRange=np.arange(0,numSecs,(1/FPS))
    
    # round down the Fx and Fy coordinates
    floorFx=np.floor(fx)
    floorFy=np.floor(fy)
    
    # make them ints
    floorFx=floorFx.astype(int)
    floorFy=floorFy.astype(int)
    
    # make heatmaps of each fish: a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(floorFx, floorFy, bins=10)
    
    # convert pixels to mm for fx,fy,bx,by,ex and ey
    [fx_mm,bx_mm,ex_mm],[fy_mm,by_mm,ey_mm] = AZU.convertToMm([fx,bx,ex],[fy,by,ey]) 
    
    # Compute distance travelled between each frame 
    distPerFrame,cumDist=AZU.computeDistPerFrame(bx_mm,by_mm)

    # Check length, looking for shortest in the list over the minimum
    AZU.checkTracking(distPerFrame)
    avgVelocity=cumDist[-1] /(len(cumDist)/FPS)   # per second over whole movie
    
    # Compute bouts and measure BPS, heading change per bout, left turn percent. excise all bouts, and take the average bout (with SD)
    idF=[]
    if(createFigures):
        idF=figureFolder
    distPerSec=distPerFrame*FPS            
    ret,RTurns,LTurns,FSwims,BPS, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC,boutStarts,boutEnds,_,_ = AZA.extractBouts(fx_mm,fy_mm, ort,distPerSec,name=name, savepath=idF,plot=True)        

    # check bouts for silly measurements
    amm=np.where((np.max(allBoutsDist,axis=1))>200)
    bmm=np.where((np.max(allBoutsDist,axis=1))<3) # exclude any that do not travel faster than 3mm/s
    keep=np.ones(len(boutAngles))
    keep[amm]=0
    keep[bmm]=0
    keep=keep>0
    
    RTurns=RTurns[keep]
    LTurns=LTurns[keep]
    FSwims=FSwims[keep]
    allBouts = allBouts[keep]
    allBoutsDist = allBoutsDist[keep]
    allBoutsOrt = allBoutsOrt[keep]
    boutAngles = boutAngles[keep]
    boutStarts = boutStarts[keep]
    boutEnds = boutEnds[keep]
    
    avgBout = np.mean(allBoutsDist,0)
    avgBoutSD = np.std(allBoutsDist,0)/np.sqrt(len(allBoutsDist))
    biasLeftBout = (np.sum(boutAngles))/(np.sum(np.abs(boutAngles))) # positive is bias for left, negative bias for right
    avgAngVelocityBout = np.mean(np.abs(boutAngles))
    # Compute bout amplitudes from all bouts peak
    boutAmps=AZA.findBoutMax(allBoutsDist)
#    OR
    # Compute boutAmps from integral of distance travelled during that bout
#    boutDists=AZA.findBoutArea(allBoutsDist)
    
    boutDists,_=AZU.computeDistPerBout(bx_mm,by_mm,boutStarts,boutEnds)
    if omitForward:
        boutKeep,boutSeq=AZP.angleToSeq_LR(boutAngles)
        comb1,seqProbs1=AZP.probSeq1(boutSeq,pot=['L','R']) # compute overall observed probability of each bout type (LR)
        comb2,randProbs2, seqProbs2_V, seqProbs2_Z, seqProbs2_P = AZP.probSeq2(boutSeq,pot=['L','R']) # compute the observed probability of each bout type (FLR) appearing in pairs. Returns labels, raw probabilities and normalised probabilities (relative probability from expected)
    else:
        boutSeq=AZP.angleToSeq(boutAngles)
        comb1,seqProbs1=AZP.probSeq1(boutSeq,pot=['F','L','R']) # compute overall observed probability of each bout type (FLR)
        comb2,randProbs2, seqProbs2_V, seqProbs2_Z, seqProbs2_P = AZP.probSeq2(boutSeq,pot=['F','L','R']) # compute the observed probability of each bout type (FLR) appearing in pairs. Returns labels, raw probabilities and normalised probabilities (relative probability from expected)
    
    # Compute the cumulative angle over the movie
    # avgAngVelocity,bias,cumOrt=AZA.computeCumulativeAngle(ort,plot=False)
#    AZA.boutHeadings(ort, boutStarts, boutStops)
    
    params=[]
    
#    name=name+pstr
    params.append(  date                )
    params.append(  gType               )
    params.append(  cond                )
    params.append(  chamber             )
    params.append(  fishNo              )
    params.append(  trackingFile        )
    params.append(  AZU.grabAviFileFromTrackingFile(trackingFile))
    params.append(  BPS                 )
    params.append(  avgVelocity         )
    params.append(  distPerFrame        )
    params.append(  cumDist             )
    params.append(  avgBout             )
    params.append(  avgBoutSD           )
    params.append(  boutAmps            )
    params.append(  boutDists           )
    params.append(  boutAngles          )
    params.append(  heatmap             )
    params.append(  avgAngVelocityBout  )
    params.append(  biasLeftBout        )
    params.append(  LturnPC             )
    params.append(  boutSeq             )
    params.append(  seqProbs1           )
    params.append(  seqProbs2_V         )
    params.append(  seqProbs2_P         )
    params.append(  seqProbs2_Z         )
    params.append(  boutStarts          )
    
    thisFishDict=AZS.populateSingleDictionary(params=params,allBoutsList=allBouts,allBoutsOrtList=allBoutsOrt,allBoutsDistList=allBoutsDist,comb1=comb1,comb2=comb2)
    
    
    thisFishDictName=DictionaryFolder+name+'_ANALYSIS' + '.npy'
    if(createDict):np.save(dictOutNames[k],thisFishDict) 
    dictList.append(thisFishDict)
    print('Analysis saved at ' + thisFishDictName)
    
    if(createFigures):
        print('Saving figures at ' + figureFolder)
        AZU.tryMkDir(figureFolder)
        AZU.tryMkDir(figureFolder+'avgBout\\')
        AZU.tryMkDir(figureFolder+'HeatMaps\\')
        AZU.tryMkDir(figureFolder+'CumDist\\')
        AZU.tryMkDir(figureFolder+'boutAmps\\')
        
        shtarget=AZF.indAvgBoutFig(avgBout,avgBoutSD,name,figureFolder+'avgBout\\')
        
#        if(shtarget!=-1):
#            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
#        else:
#            print('WARNING!! Saving figure' + name + '_avgBout failed')
        ##    
        shtarget=AZF.indHeatMapFig(heatmap,name,figureFolder+'HeatMaps\\')
        
#        if(shtarget!=-1):
#            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
#        else:
#            print('WARNING!! Saving figure' + name + '_heatmap failed')
        ##
        if(np.sum(cumDist)!=0):
            shtarget=AZF.indCumDistFig(cumDist,name,figureFolder+'CumDist\\')
        else:
            shtarget=-1
            
#        if(shtarget!=-1):
#            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
#        else:
#            print('WARNING!! Saving figure' + name + '_cumDist failed')
        if(np.sum(boutAmps)!=0):
           shtarget=AZF.indBoutAmpsHistFig(boutAmps,name,figureFolder+'boutAmps\\')
        else:
            shtarget=-1
#        if(shtarget!=-1):
#            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
#        else:
#            print('WARNING!! Saving figure' + name + '_cumDist failed')
        
        if(keepFigures==False):plt.close('all')
        
    ##### END OF FILE LOOP #######
#FIN
