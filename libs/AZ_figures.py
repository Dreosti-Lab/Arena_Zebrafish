# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:20:07 2020

@author: thoma
"""

# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'

import sys
sys.path.append(lib_path)
import numpy as np
import matplotlib.pyplot as plt
import AZ_utilities as AZU
import AZ_analysis as AZA
import AZ_video as AZV
import scipy.stats as stats
import AZ_streakProb as AZP
import AZ_math as AZM
#from itertools import compress
import cv2
from matplotlib.lines import Line2D
import glob
#import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr
#Rstats = importr('stats')

def run(g1,g2,l1,l2,savepath=r'D:\\Shelf\\',keepFigures=False,save=True,recompute=False,startFrame=0,endFrame=432000,gridSize=40,SO=False,dicFolder=r'D:\Analysis\GroupedData\Dictionaries\\'):
    
    savepath=savepath+r'\\Figures\\'
#    g1='EmxGFP_B0_200913'
#    g2='WT_M0_200826'
#    l1='Blank'
#    l2='Maze'
#    save=True
    dic1File=dicFolder+g1 +'.npy'
    dic2File=dicFolder+g2+'.npy'
    dicList=[]
    dicList.append(dic1File)
    dicList.append(dic2File)
    print('Running turn figures')
    runTurnFigs(dicList,l1,l2,savepath=savepath,keepFigures=keepFigures,save=save)
    print('Running dispersal figures')
    GroupStateProps=dispersalFigures(dicList,l1,l2,savepath=savepath,saveFigures=save,recomputeDispersal=recompute,SO=SO,dicFolder=dicFolder)
    print('Running heatmaps')
    spatialMaps(dicList,savepath=savepath,startFrame=startFrame,endFrame=endFrame,gridSize=gridSize,save=save,keepFigures=keepFigures)
    return GroupStateProps

def runTurnFigs(dicList,label1,label2,savepath=r'D:\\Shelf\\',keepFigures=True,save=True): # only works for 2 dictionaries in a list at present
#    label1='1st30min'
#    label2='2nd30min'
    savepath=savepath+r'\\TurnFigs\\'
    AZU.cycleMkDir(savepath)
    dic1=dicList[0]
    dic2=dicList[1]
    dict1=np.load(dic1,allow_pickle=True).item()
    dict2=np.load(dic2,allow_pickle=True).item()
    n1,n2=turnTriggeredAngleHist(dict1,dict2,savepath=savepath,label1=label1,label2=label2,keepFigures=keepFigures,saveFigures=save)
    LRChainAnalysis(dict1,dict2,savepath=savepath,label1=label1,label2=label2,keepFigures=keepFigures,saveFigures=save)
    FLRBoutCompare(dict1,dict2,savepath=savepath,label1=label1,label2=label2,keepFigures=keepFigures,saveFigures=save)
    return n1,n2

def plotLoomTrajectoryFolder(trackingFolder):
    trackingFiles=[]
    trackingsubFiles = glob.glob(trackingFolder + r'\*.npz')
    for s in trackingsubFiles:trackingFiles.append(s)
    for trackingFile in trackingFiles:
        plotLoomTrajectories(trackingFile)
        
def plotLoomTrajectories(trackingFile):
    
    fx,fy,_,_,_,_,_,heading,_=AZU.grabTrackingFromFile(trackingFile)
    
    loomStarts,loomEnds=AZA.findLooms(len(fx),
                                      startTime=15,
                                      interval=1,
                                      duration=1,
                                      numFrames=120,
                                      frameRate=120)

    trajX,trajY,trajHeadings,trajXOrig,trajYOrig,=AZA.extractTrajFromStim(loomStarts,loomEnds,fx,fy,heading)
    rotTrajX,rotTrajY=AZA.rotateTrajectoriesByHeadings(trajXOrig,trajYOrig,trajHeadings)
    
    plt.figure()
    for i in range(0,len(rotTrajX)):
        plt.plot(rotTrajX[i],rotTrajY[i])
        plt.ylim(-500,500)
        plt.xlim(-500,500)
        
    return rotTrajX,rotTrajY,trajHeadings,trajX,trajY

def defineStateFromDispersal(dispVec,states,colList,ExploreThreshMean=9.6,ExploreThreshSD=2.5,ExploitThreshMean=2.3,ExploitThreshSD=1.3):

# Define exploration vs exploitation (exploration state (Exploration: "...9.6 ± 2.5 mm, mean ± s.d...."; Exploitation:"... 2.3 ± 1.3 mm as per Marques et al Nature 2020, Robson and Li labs)
    # label different parts based on absolute threshold : this will vary a little according to fish but we will fix this later when we know more about escape states across fish and absolute maximum velocity. First we will use the same as the results of Li and Robson's HMM model)
    ExploreLowThresh=ExploreThreshMean-(ExploreThreshSD*3)
    ExploreHighThresh=ExploreThreshMean+(ExploreThreshSD*3)
    
    ExploitLowThresh=ExploitThreshMean-(ExploitThreshSD*3)
    ExploitHighThresh=ExploitThreshMean+(ExploitThreshSD*3)
    
    if ExploitHighThresh>ExploreLowThresh: # if the threshold overlap, take the weighted (by ratio of SD) average of the two and set a hard border (fudge)
        thresh=np.average([ExploitThreshMean,ExploreThreshMean],weights=[ExploreThreshSD,ExploitThreshSD])
        ExploitHighThresh=thresh
        ExploreLowThresh=thresh
    
    if ExploitLowThresh<0.1: ExploitLowThresh=0.1
    
    fishState=[]
    fishCol=[]
    for k in range(len(dispVec)):
        # use color labels and list the state
        if dispVec[k]<ExploitLowThresh: # if dispersal is under the lowest thresh for exploitation, fish has 'Stopped'
            fishState.append(states[0])
            fishCol.append(colList[0])
        elif dispVec[k] >= ExploitLowThresh and dispVec[k] <= ExploitHighThresh: # if dispersal is within Exploit range, fish is 'Exploiting'
            fishState.append(states[1])
            fishCol.append(colList[1])
        elif dispVec[k] >= ExploreLowThresh and dispVec[k] <= ExploreHighThresh: # if dispersal is within Explore range, fish is 'Exploring'
            fishState.append(states[2])
            fishCol.append(colList[2])
        elif dispVec[k] > ExploreHighThresh: # if dispersal is higher than Explore range, fish is 'Escaping'
            fishState.append(states[3])
            fishCol.append(colList[3])
            
    return fishState, fishCol, ExploreHighThresh

def dispersalFigures(dictList,l1,l2,savepath=r'D:\\Shelf\\',templateDir=r'D:\\Templates\\',startFrame=0,endFrame=432000,smoothWindow=120,dispWindow=5,frameRate=120,saveFigures=True,saveDispersal=True,recomputeDispersal=False,SO=True,dicFolder='D:\\Analysis\\GroupedData\\Dictionaries\\'):
    
    savepath=savepath+r'\\DispersalFigs\\'
    # list possible states
    states=[]
    states.append('Stop')
    states.append('Exploit')
    states.append('Explore')
    states.append('Escape')
    
    # list each state's color
    colList=[]
    colList.append('Black')
    colList.append('Green')
    colList.append('Orange')
    colList.append('Magenta')
    
    groupNames=[]
    GroupStatePropsS=[] # to store relative time proportion in each state for each group
    meanDispS=[]
    SDDispS=[]
    # cycle through dictionaries
    for i,f in enumerate(dictList): # cycle through group dictionaries   
        print('Loading group dictionary...')
        # load dictionary and find names and numbers
        if SO:
            f=dicFolder+f+'.npy'
            
        dic=np.load(f,allow_pickle=True).item()
        groupNames.append(dic['Name'])
        numFish=len(dic['Ind_fish'])
        GroupStateProps=[]
        fishStateProps=[]
        meanDisp=[]
        SDDisp=[]
        # cycle through fish
        for j in range(numFish): # cycle through individual fish 
            # find fish
            thisFish=dic['Ind_fish'][j]
            fishname=thisFish['info']['AviPath']
            fishname=fishname[4:-4]
            print('Loaded individual fish ' + fishname)
            # find avi
            fishname=glob.glob(templateDir+r'\\'+fishname+'*.avi')
            fishname=fishname[0]
            img=AZU.grabFrame(fishname,0)
            
            # find tracking
            fx,fy,_,_,_,_,_,_,_ = AZU.grabTrackingFromFile(thisFish['info']['TrackingPath'])
            
            # measure dispersal over time (5 second window as in Marques et al Nature 2020 (Robson and Li labs)) 
            
            # convert to mm
            fx_mm,fy_mm=AZU.convertToMm(fx,fy)
            
            # check to see if dispVec has been done already
            flag=False
            if 'dispersal' in thisFish['data'] :
                flag=True
                if (recomputeDispersal):
                    print('Dispersal already measured, but recompute dispersal turned ON. Overwriting...')
                else: 
                    print('Dispersal already measured for this fish, grabbing...')
                    dispVec=thisFish['data']['dispersal'][startFrame:endFrame]
            
            if (flag and recomputeDispersal) or flag==False: 
                dispVec=AZA.measureDispersal(fx_mm,fy_mm, window=dispWindow)
            
                if saveDispersal:
                    thisFish['data']['dispersal']=dispVec
                    dic['Ind_fish'][j]=thisFish
                    print('Overwriting dictionary at ' + f)
                    np.save(f,dic)
                    
                dispVec=dispVec[startFrame:endFrame]
                
            dispVec_sm=AZM.smoothSignal(dispVec,smoothWindow)
            fx=fx[startFrame:endFrame]
            fy=fy[startFrame:endFrame]
            fx_mm=fx_mm[startFrame:endFrame]
            fy_mm=fy_mm[startFrame:endFrame]
            fishState,fishCol, ExploreHighThresh=defineStateFromDispersal(dispVec,states,colList,ExploreThreshMean=9.6,ExploreThreshSD=2.5,ExploitThreshMean=2.3,ExploitThreshSD=1.3)
            fishState_sm,fishCol_sm,_=defineStateFromDispersal(dispVec_sm,states,colList,ExploreThreshMean=9.6,ExploreThreshSD=2.5,ExploitThreshMean=2.3,ExploitThreshSD=1.3)
            
            # Compute total proportion time for this fish in different states
            thisFishProps=[]
            for thisState in states:
                prop=[]
                for n in fishState_sm:
                    prop.append(n==thisState)
                propSum=np.sum(prop)
                thisFishProps.append(np.divide(propSum,len(fishState_sm)))
            fishStateProps.append(thisFishProps) # collect all relative times for this fish
                                
            if saveFigures:
                aa=fishname.rsplit('\\',2)
                # space first 
                saveDir=aa[0]+'\\DispersalFigs\\Space'
                
                figName='DispersalStateThreshMarques_Space20min'
                plt.figure(figName)
                plt.imshow(img*-1,cmap='Greys')
                plt.scatter(fx[0:144000],fy[0:144000],c=fishCol[0:144000],s=1.5)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                # Construct legend
                p0 = Line2D([0], [0], marker='o', color='w', label=states[0],markerfacecolor=colList[0],markersize=10)
                p1 = Line2D([0], [0], marker='o', color='w', label=states[1],markerfacecolor=colList[1],markersize=10)
                p2 = Line2D([0], [0], marker='o', color='w', label=states[2],markerfacecolor=colList[2],markersize=10)
                p3 = Line2D([0], [0], marker='o', color='w', label=states[3],markerfacecolor=colList[3],markersize=10)
                plt.legend(handles=[p0,p1,p2,p3],bbox_to_anchor=(1.05, 1))
                
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                if i ==0:
                    saveDirS='D:\\Shelf\\DispersalFigs\\' + l1 + '\\Space'
                else: saveDirS='D:\\Shelf\\DispersalFigs\\' + l2 + '\\Space'
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                figName='DispersalStateThreshMarques_SpaceAll'
                plt.figure(figName)
                plt.imshow(img*-1,cmap='Greys')
                plt.scatter(fx,fy,c=fishCol,s=1.5)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.legend(handles=[p0,p1,p2,p3],bbox_to_anchor=(1.05, 1))
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                plt.scatter(fx[144001:-1],fy[144001:-1],c=fishCol[144001:-1],s=1)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                # now time
                if i ==0:
                    saveDirS='D:\\Shelf\\DispersalFigs\\'+ l1 +'\\Time'
                else: saveDirS='D:\\Shelf\\DispersalFigs\\' + l2 + '\\Time'
                saveDir=aa[0]+'\\DispersalFigs\\Time'
                
                figName='DispersalStateThreshMarques_Time20mins'
                plt.figure(figName)
                xx=range(144000)
                xx=np.divide(xx,frameRate)
                xx=np.divide(xx,60)
                dispVecA=dispVec
                dispVecA[dispVec>25]=25
                plt.scatter(xx,dispVecA[0:144000],c=fishCol[0:144000],s=1)
                plt.xticks(ticks=np.linspace(0,20,num=21))
                plt.xlabel('Time (mins)')
                plt.ylabel('Dispersal (mm) (lim to 25mm)')
                plt.ylim(0,25)
                plt.legend(handles=[p0,p1,p2,p3],bbox_to_anchor=(1.05, 1))
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                figName='DispersalThreshMarques_Time'
                plt.figure(figName)
                xx=range(len(fx))
                xx=np.divide(xx,frameRate)
                xx=np.divide(xx,60)
                plt.scatter(xx,dispVecA,c=fishCol,s=1)
                plt.xlabel('Time (mins)')
                plt.ylabel('Dispersal (mm) (lim to 25mm)')
                plt.ylim(0,30)
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                figName='Dispersal'
                plt.figure(figName)
                xx=range(len(fx))
                xx=np.divide(xx,frameRate)
                xx=np.divide(xx,60)
                plt.scatter(xx,dispVecA,s=1)
                plt.xlabel('Time (min)')
                plt.ylabel('Dispersal (mm) (lim to 25mm)')
                plt.ylim(0,25)
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                # and finally distributions
                if i ==0:
                    saveDirS='D:\\Shelf\\DispersalFigs\\' + l1 + '\\Distribution'
                else: saveDirS='D:\\Shelf\\DispersalFigs\\' + l2 + '\\Distribution'
                saveDir=aa[0]+'\\DispersalFigs\\Distribution'
                
                figName='DispersalDistribution'
                plt.figure(figName)
                dispHist,c=np.histogram(dispVec,bins=100)
                c = (c[:-1]+c[1:])/2
                plt.plot(c,dispHist)
                plt.xlabel('Dispersal (mm)')
                plt.ylabel('Frequency (Time frames)')
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                figName='DispersalDistributionNorm'
                plt.figure(figName)
                dispVec_norm=np.divide(dispVec,np.max(dispVec[dispVec<ExploreHighThresh])) # exclude escapes from normalisation
                dispHist,c=np.histogram(dispVec_norm,bins=100)
                dispHist_pdf=np.divide(dispHist,np.sum(dispHist))
                c = (c[:-1]+c[1:])/2
                plt.plot(c,dispHist_pdf)
                plt.xlabel('Normalised Dispersal')
                plt.ylabel('Probability Density')
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
               
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                figName='DispersalDistributionNorm50%'
                plt.xlim(0,0.5)
#                plt.ylim(0,0.06)
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                # Distributions excluding the escape periods
                dispVec=AZU.filterBursts(dispVec)
                
                figName='DispersalDistribution_ExcBurst'
                plt.figure(figName)
                dispHist,c=np.histogram(dispVec,bins=100)
                c = (c[:-1]+c[1:])/2
                plt.plot(c,dispHist)
                plt.xlabel('Dispersal (mm)')
                plt.ylabel('Frequency (Time frames)')
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                figName='DispersalDistributionNorm_ExcBurst'
                plt.figure(figName)
                dispVec_norm=np.divide(dispVec,np.max(dispVec)) # exclude escapes from normalisation
                dispHist,c=np.histogram(dispVec_norm,bins=100)
                dispHist_pdf=np.divide(dispHist,np.sum(dispHist))
                c = (c[:-1]+c[1:])/2
                plt.plot(c,dispHist_pdf)
                plt.xlabel('Normalised Dispersal')
                plt.ylabel('Probability Density')
                saveName=saveDir+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
               
                saveName=saveDirS+'\\'+aa[2][:-4]+'_'+figName+'.png'
                AZU.cycleMkDirr(saveName)
                print('Saving images at ' + saveName)
                plt.savefig(saveName,dpi=600)
                plt.close()
                
                 
                saveDir=aa[0]+'\\DispersalFigs\\StateProportions'
                
            # collect proportions in each state for each group
            GroupStateProps.append(fishStateProps)
            meanDisp.append(np.mean(dispVec_sm))
            SDDisp.append(np.std(dispVec_sm))
            ### END OF FISH LOOP ###
        GroupStatePropsS.append(GroupStateProps)
        meanDispS.append(meanDisp)
        SDDispS.append(SDDisp)
        ### END OF DICT LOOP ###
    return GroupStatePropsS,meanDispS,SDDisp  
               
                
            
def spatialMaps(dictList,savepath=r'D:\\Shelf\\',startFrame=0,endFrame=60*60*120,gridSize=40,save=True,keepFigures=False):
    
    savepath=savepath+r'\\spatialFigs\\'
    if save==False and keepFigures==False : print('Not saving or keeping figures...I guess you are debugging...?')
        
    # Maps for groups
    boutCountSS=[]
    boutDensitySS=[]
    distSumSS=[]
    distDensitySS=[]
    distAverageSS=[]
    angleSumSS=[]
    angleDensitySS=[]
    angleAverageSS=[]
    absangleSumSS=[]
    absangleDensitySS=[]
    absangleAverageSS=[]
    dispersalAverageSS=[]
    dispersalDensitySS=[]
    pntCentreNormSS=[]
    pntSurroundNormSS=[]
    groupNames=[]
    
    ## START OF GROUP LOOP
    for i,f in enumerate(dictList): # cycle through group dictionaries   
        # load dictionary and find names and numbers
        dic=np.load(f,allow_pickle=True).item()
        groupNames.append(dic['Name'])
        numFish=len(dic['Ind_fish'])
        # Maps for fish
        boutCountS=[]
        boutDensityS=[]
        distSumS=[]
        distDensityS=[]
        distAverageS=[]
        angleSumS=[]
        angleDensityS=[]
        angleAverageS=[]
        absangleSumS=[]
        absangleDensityS=[]
        absangleAverageS=[]
        dispersalAverageS=[]
        dispersalDensityS=[]
        pntCentreNormS=[]
        pntSurroundNormS=[]
        
        ## START OF FISH LOOP
        for j in range(numFish):# cycle through individual fish 
            if (i==0 and j!=11) or (i==1 and j!=7):
                print(j)
                # find fish
                thisFish=dic['Ind_fish'][j]
               
                # find avi
                vid=cv2.VideoCapture(thisFish['info']['AviPath'])
                w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # find width and height
                vid.release()
                
                # find tracking
                fx,fy,_,_,_,_,_,_,_ = AZU.grabTrackingFromFile(thisFish['info']['TrackingPath'])
                # crop out start and endframe (whole tracking by default)
                fx=fx[startFrame:endFrame]
                fy=fy[startFrame:endFrame]
               
                
                
                boutStarts=thisFish['data']['boutStarts'] # extract boutStarts
                
                # extract the following features to make maps:
                boutDists=thisFish['data']['boutDists'][boutStarts<len(fx)] # boutDists
                boutAngles=thisFish['data']['boutAngles'][boutStarts<len(fx)] # boutAngles    
                boutStarts=boutStarts[boutStarts<len(fx)]
                boutDispersal=thisFish['data']['dispersal'][boutStarts]
                
                # make maps for this fish
                boutCount,boutDensity,distSum,distDensity,distAverage = AZA.heatmapFeature(fx,fy,boutStarts,boutDists,imageWidth=w,imageHeight=h,gridSize=gridSize)
                _,_,angleSum,angleDensity,angleAverage = AZA.heatmapFeature(fx,fy,boutStarts,boutAngles,imageWidth=w,imageHeight=h,gridSize=gridSize)
                _,_,absAngleSum,absAngleDensity,absAngleAverage = AZA.heatmapFeature(fx,fy,boutStarts,np.abs(boutAngles),imageWidth=w,imageHeight=h,gridSize=gridSize)
                _,_,dispSum,dispDensity,dispAverage = AZA.heatmapFeature(fx,fy,boutStarts,boutDispersal,imageWidth=w,imageHeight=h,gridSize=gridSize)
                pntCentreNorm,pntSurroundNorm=AZA.computeTimeCentreSurround(fx,fy,imageWidth=w,imageHeight=h,gridSize=40)
                
                if save or keepFigures:
                    # Make figures for this fish
                    saveDir='D:\\Shelf\\IndHeatMaps\\'
                    # bout density
                    fName=thisFish['info']['Date']+'_'+ thisFish['info']['Genotype']+'_'+thisFish['info']['Condition']+'_'+thisFish['info']['Chamber']+'_'+thisFish['info']['FishNo']
                    figName=fName + '_boutCount'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(boutCount,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
            #        plt.clim(0,0.015)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # bout Density
                    figName=fName + '_boutDensity'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(boutDensity,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
    #                plt.clim(0,0.015)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # distAverage
                    figName=fName + '_distAverage'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(distAverage,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
    #                plt.clim(0,2)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # distDensity
                    figName=fName + '_distDensity'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(distDensity,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
            #        plt.clim(0,1)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # absangleDensity
                    figName=fName + '_absangleDensity'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(absAngleDensity,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
            #        plt.clim(0,0.015)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # absangleAverage
                    figName=fName + '_absangleAverage'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(absAngleAverage,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
            #        plt.clim(0,30)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # dispersalAverage
                    figName=fName + '_dispersalAverage'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(dispAverage,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
            #        plt.clim(0,1)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # dispersalDensity
                    figName=fName + '_dispersalDensity'
                    plt.figure(figName)
                    plt.title(figName)
                    plt.imshow(dispDensity,cmap='gist_heat')
                    plt.xlim(0.5,gridSize-0.5)
                    plt.ylim(0.5,gridSize-0.5)
    #                plt.clim(0,0.004)
                    plt.colorbar()
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    saveName=saveDir+figName+'.png'
                    AZU.cycleMkDirr(saveName)
                    if save:
                        print('Saving figure as ' + saveName)
                        plt.savefig(saveName,dpi=600)
                    if keepFigures==False:
                        plt.close()
                    
                    # collect fish for this group
                    boutCountS.append(boutCount)    
                    boutDensityS.append(boutDensity)
                    distSumS.append(distSum)
                    distDensityS.append(distDensity)
                    distAverageS.append(distAverage)
                    angleSumS.append(angleSum)
                    angleDensityS.append(angleDensity)
                    angleAverageS.append(angleAverage)
                    absangleSumS.append(absAngleSum)
                    absangleDensityS.append(absAngleDensity)
                    absangleAverageS.append(absAngleAverage)
                    dispersalAverageS.append(dispAverage)
                    dispersalDensityS.append(dispDensity)
                    pntCentreNormS.append(pntCentreNorm)
                    pntSurroundNormS.append(pntSurroundNorm)
                
                # collect groups
                pntCentreNormSS.append(pntCentreNormS)
                pntSurroundNormSS.append(pntSurroundNormS)
            ## END OF FISH LOOP
            
        boutCountSS.append(boutCountS)
        boutDensitySS.append(boutDensityS)
        distSumSS.append(distSumS)
        distDensitySS.append(distDensityS)
        distAverageSS.append(distAverageS)
        angleSumSS.append(angleSumS)
        angleDensitySS.append(angleDensityS)
        angleAverageSS.append(angleAverageS)
        absangleSumSS.append(absangleSumS)
        absangleDensitySS.append(absangleDensityS)
        absangleAverageSS.append(absangleAverageS)
        dispersalAverageSS.append(dispersalAverageS)
        dispersalDensitySS.append(dispersalDensityS)
        ## END OF GROUP LOOP
        
    figName='ProportionTimeSpentInCentre'
    plt.figure(figName)
    plt.title(figName)
    groupCentreAv=[]
    groupCentreSE=[]
    
    for gI in range(len(groupNames)):
        
        # Time spent in centre
        index = np.arange(len(groupNames))
        numFish=len(pntCentreNormSS[gI])
        
        # find average and se of each group
        groupCentreAv.append(np.mean(pntCentreNormSS[gI]))
        SD=np.std(pntCentreNormSS[gI])
        groupCentreSE.append(np.divide(SD,np.sqrt(numFish)))
        
    # plot line between means with SE bars
    plt.plot(index,groupCentreAv)
    plt.errorbar(index,groupCentreAv,yerr=groupCentreSE)
    
    # plot all points from each fish (one per fish)
    for gI in range(len(groupNames)):
        for j in range(numFish):
            y=pntCentreNormSS[gI][j]
            x=np.random.normal(index[gI], 0.08)
            plt.scatter(x,y,alpha=0.6,color='black',s=4)
    
    plt.title(figName)
    plt.xticks(ticks=index, labels=groupNames)
    saveDir='D:\\Shelf\\'
    saveName=saveDir+figName+groupNames[0] + 'vs' + groupNames[1]+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName,dpi=600)
    plt.close()
        
    for gI,gName in enumerate(groupNames):
        # bout Count
        thisgroup_boutCountAVG=np.array(boutCountSS[gI]).mean(axis=0)
        figName=gName + '_boutDensityAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_boutCountAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
#        plt.clim(0,0.015)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName,dpi=600)
        plt.close()
        
        # bout Density
        thisgroup_boutDensityAVG=np.array(boutDensitySS[gI]).mean(axis=0)
        figName=gName + '_boutDensityAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_boutDensityAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
#        plt.clim(0,0.015)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # distDensity
        thisgroup_distDensityAVG=np.array(distDensitySS[gI]).mean(axis=0)
        figName=gName + '_distDensityAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_distDensityAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
#        plt.clim(0,1)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # distAverage
        thisgroup_distAverageAVG=np.array(distAverageSS[gI]).mean(axis=0)
        figName=gName + '_distAverageAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_distAverageAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
        plt.clim(0,1.5)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # absangleDensity
        thisgroup_absangleDensityAVG=np.array(absangleDensitySS[gI]).mean(axis=0)
        figName=gName + '_absangleDensityAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_absangleDensityAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
#        plt.clim(0,0.015)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # absangleAverage
        thisgroup_absangleAverageAVG=np.array(absangleAverageSS[gI]).mean(axis=0)
        figName=gName + '_absangleAverageAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_absangleAverageAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
        plt.clim(0,40)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # Dispersal density
        thisgroup_dispDensityAVG=np.array(dispersalDensitySS[gI]).mean(axis=0)
        figName=gName + '_dispersalDensityAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_dispDensityAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
        plt.clim(0,0.004)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
        # dispersalAverage
        thisgroup_dispAverageAVG=np.array(dispersalAverageSS[gI]).mean(axis=0)
        figName=gName + '_dispersalAverageAVG'
        plt.figure(figName)
        plt.title(figName)
        plt.imshow(thisgroup_dispAverageAVG,cmap='gist_heat')
        plt.xlim(0.5,gridSize-0.5)
        plt.ylim(0.5,gridSize-0.5)
        plt.clim(0,8)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        saveDir='D:\\Shelf\\GroupHeatMaps\\'
        saveName=saveDir+figName+'.png'
        AZU.cycleMkDirr(saveName)
        print('Saving figure as ' + saveName)
        plt.savefig(saveName)
        plt.close()
        
    # distAverageDiff
    g0=np.array(distAverageSS[0]).mean(axis=0)
    g1=np.array(distAverageSS[1]).mean(axis=0)
    distDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_distAverageDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(distDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # distAverageDensityDiff
    g0=np.array(distDensitySS[0]).mean(axis=0)
    g1=np.array(distDensitySS[1]).mean(axis=0)
    distDensDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_distDensityDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(distDensDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # dispersalAverageDiff
    g0=np.array(dispersalAverageSS[0]).mean(axis=0)
    g1=np.array(dispersalAverageSS[1]).mean(axis=0)
    dispDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_dispersalAverageDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(dispDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # dispersalAverageDensityDiff
    g0=np.array(dispersalDensitySS[0]).mean(axis=0)
    g1=np.array(dispersalDensitySS[1]).mean(axis=0)
    dispDensDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_dispersalDensityDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(dispDensDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # boutCountDiff
    g0=np.array(boutCountSS[0]).mean(axis=0)
    g1=np.array(boutCountSS[1]).mean(axis=0)
    boutCountDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_boutCountDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(boutCountDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # boutDensDiff
    g0=np.array(boutDensitySS[0]).mean(axis=0)
    g1=np.array(boutDensitySS[1]).mean(axis=0)
    boutDensDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_boutDensDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(boutDensDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # boutCountSDDiff
    g0=np.array(boutCountSS[0]).std(axis=0)
    g1=np.array(boutCountSS[1]).std(axis=0)
    boutCountSDDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_boutCountSDDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(boutCountSDDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()
    
    # boutDensSDDiff
    g0=np.array(boutDensitySS[0]).std(axis=0)
    g1=np.array(boutDensitySS[1]).std(axis=0)
    boutDensitySDDiff=np.subtract(g1,g0)
    figName=groupNames[0] + '-' + groupNames[1] +'_boutDensitySDDiff'
    plt.figure(figName)
    plt.title(figName)
    plt.imshow(boutDensitySDDiff,cmap='RdBu_r')
    plt.xlim(0.5,gridSize-0.5)
    plt.ylim(0.5,gridSize-0.5)
#    plt.clim(0,1)
    plt.colorbar()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    saveDir='D:\\Shelf\\GroupHeatMaps\\'
    saveName=saveDir+figName+'.png'
    AZU.cycleMkDirr(saveName)
    print('Saving figure as ' + saveName)
    plt.savefig(saveName)
    plt.close()

def turnTriggeredAngleHist(dic1,dic2,savepath=r'D:\\Shelf\\',saveFigures=True,keepFigures=False,label1='NO LABEL!',label2='NO LABEL!'):
   
    name1=dic1['Name']
    name2=dic2['Name']
    
    angles1=dic1['PooledData']['boutAngles']
    angles2=dic2['PooledData']['boutAngles']
    
    n1=len(angles1)
    n2=len(angles2)
    
    allAngles1=[item for sublist in angles1 for item in sublist]
    allAngles2=[item for sublist in angles2 for item in sublist]
    
    boo1,boutSeq_1=AZP.angleToSeq_LR(allAngles1)
    boo2,boutSeq_2=AZP.angleToSeq_LR(allAngles2)

    rightTrigAngle1=[]
    leftTrigAngle1=[]
    rightTrigAngle2=[]
    leftTrigAngle2=[]
    
    for i,t in enumerate(boutSeq_1):
        if i==len(boutSeq_1)-1:break
        # find this bout in the original angles list
        countTurns=-1
        toDo=True
        for j,k in enumerate(boo1):
            if toDo:
                if k: 
                    countTurns+=1
                if countTurns==i and t=='R':
                    rightTrigAngle1.append(allAngles1[j+1])
                    toDo=False
                    break
                if countTurns==i and t=='L':
                    leftTrigAngle1.append(allAngles1[j+1])
                    toDo=False
                    break
            else: break

    for i,t in enumerate(boutSeq_2):
        if i==len(boutSeq_2)-1:break
        # find this bout in the original angles list
        countTurns=-1
        toDo=True
        for j,k in enumerate(boo2):
            if toDo:
                if k: 
                    countTurns+=1
                if countTurns==i and t=='R':
                    rightTrigAngle2.append(allAngles2[j+1])
                    toDo=False
                    break
                if countTurns==i and t=='L':
                    leftTrigAngle2.append(allAngles2[j+1])
                    toDo=False
                    break
            else: break
        
    leftHist1,cL1=np.histogram(leftTrigAngle1,bins=90)
    leftHist2,cL2=np.histogram(leftTrigAngle2,bins=90)
    rightHist1,cR1=np.histogram(rightTrigAngle1,bins=90)
    rightHist2,cR2=np.histogram(rightTrigAngle2,bins=90)
    
    leftHist1_norm=leftHist1/np.sum(leftHist1)
    leftHist2_norm=leftHist2/np.sum(leftHist2)
    rightHist1_norm=rightHist1/np.sum(rightHist1)
    rightHist2_norm=rightHist2/np.sum(rightHist2)
    
    ccL1 = (cL1[:-1]+cL1[1:])/2
    ccL2 = (cL2[:-1]+cL2[1:])/2
    ccR1 = (cR1[:-1]+cR1[1:])/2
    ccR2 = (cR2[:-1]+cR2[1:])/2
    
    figName='Turn triggered angle histogram' + name1 + '_' + label1
    plt.figure(figName)
    plt.plot(ccL1*-1,leftHist1_norm,color='blue',label='Left triggered')
    plt.plot(ccR1*-1,rightHist1_norm,color='red',label='Right triggered')#
       
    plt.title(figName)
    plt.xlim(-100,100)
    plt.xlabel('Angle')
    plt.ylabel('Relative Frequency')
    plt.ylim(0,0.15)
    plt.legend()
    
    saveName=savepath+ 'TurnTrigHist_' + name1 + '_' + label1
    if saveFigures: plt.savefig(saveName,dpi=600)
    if keepFigures==False: plt.close()
    
    figName='Turn triggered angle histogram' + name2 + '_' + label2
    plt.figure(figName)
    plt.plot(ccL2*-1,leftHist2_norm,color='green',label='Left triggered')
    plt.plot(ccR2*-1,rightHist2_norm,color='magenta',label='Right triggered')
       
    plt.title(figName)
    plt.xlim(-100,100)
    plt.xlabel('Angle')
    plt.ylabel('Relative Frequency')
    plt.ylim(0,0.15)
    plt.legend()
    
    saveName=savepath+ 'TurnTrigHist_' + name2 + '_' + label2 + '.png'
    if saveFigures: plt.savefig(saveName,dpi=600)
    if keepFigures==False: plt.close()
    return n1,n2
    
def LRChainAnalysis(dic1,dic2,savepath=r'D:\\Shelf\\',saveFigures=True,keepFigures=False,label1='NO LABEL!',label2='NO LABEL!',col1='#486AC6',col2='#F3930C',LTurnThresh=[40,60]):
    
    labelRand='Biased "coin flip"'
#    labelCoin='Coin flip'
    
    name1=dic1['Name']
    name2=dic2['Name']
    
    numIter=100
    numIter1=1000
    cumProbS_1=[]
    cumProbS_2=[]
    cumProbS_rand1=[]
    cumProbS_rand2=[]
    biasedCoinFlipDiff_1=[]
    biasedCoinFlipDiff_2=[]
    num1=len(dic1['PooledData']['boutAngles'])
    num2=len(dic2['PooledData']['boutAngles'])
    
    pot=['L','R']
    cumProbS_coin=[]
    for i in range(0,numIter1):
        coinSeq=[]
        for j in range(2000):
            aa=np.random.permutation(pot)
            coinSeq.append(aa[0])
        cumProbS_coin.append(AZP.probStreak_L_OR_R(coinSeq))
#    avgCumProb_coin=np.mean(cumProbS_coin,axis=0)
#    seCumProb_coin=np.std(cumProbS_coin,axis=0)/np.sqrt(numIter1)
    
#    pos_coin=avgCumProb_coin+seCumProb_coin
#    neg_coin=avgCumProb_coin-seCumProb_coin
    
    # Exclude those with LTurn percentages above or below a threshold
        # Loop through ind fish and grab the LTurnPCs
    excBoo=[]
    for i in range(num1):    
        LTurnPC=dic1['Ind_fish'][i]['data']['LTurnPC']
        # create a boolean array of those within thresholds
        if LTurnPC<LTurnThresh[0]:excBoo.append(False)
        elif LTurnPC>LTurnThresh[1]:excBoo.append(False)
        else: excBoo.append(True)
        
    for i in range(num1):
        if excBoo[i]:
            _,boutSeq_1=AZP.angleToSeq_LR(dic1['PooledData']['boutAngles'][i])
            # create random sequences to generate biased 'coin flip' probabilities for this fish        
            randProbS_1=[]

            for j in range(numIter):
                # shuffle the sequence randomly
                randBoo_1=np.random.permutation(boutSeq_1)
                randProbS_1.append(AZP.probStreak_L_OR_R(randBoo_1))
                
            # take mean for biased coin flip curve - add to list
            rP=np.mean(randProbS_1,axis=0)
            cumProbS_rand1.append(rP)
            # find real curve for this fish - add to list
            cP=AZP.probStreak_L_OR_R(boutSeq_1)
            cumProbS_1.append(cP)
            # comput difference between biased coin flip curve and real curve - add to list
            biasedCoinFlipDiff_1.append(np.subtract(rP,cP))    
    
    # Exclude those with LTurn percentages above or below a threshold
        # Loop through ind fish and grab the LTurnPCs
    excBoo=[]
    for i in range(num2):    
        LTurnPC=dic2['Ind_fish'][i]['data']['LTurnPC']
        # create a boolean array of those within thresholds
        if LTurnPC<LTurnThresh[0]:excBoo.append(False)
        elif LTurnPC>LTurnThresh[1]:excBoo.append(False)
        else: excBoo.append(True)
        
    for i in range(num2):
        if excBoo[i]:
            _,boutSeq_2=AZP.angleToSeq_LR(dic2['PooledData']['boutAngles'][i])    
            randProbS_2=[]
            for j in range(numIter):
                randBoo_2=np.random.permutation(boutSeq_2)
                randProbS_2.append(AZP.probStreak_L_OR_R(randBoo_2))
        
            rP=np.mean(randProbS_2,axis=0)
            cumProbS_rand2.append(rP)
            cP=AZP.probStreak_L_OR_R(boutSeq_2)
            cumProbS_2.append(cP)
            biasedCoinFlipDiff_2.append(np.subtract(rP,cP))
    
    avgBiasedCoinFlipDiff_1=np.mean(biasedCoinFlipDiff_1,axis=0)
    avgBiasedCoinFlipDiff_2=np.mean(biasedCoinFlipDiff_2,axis=0)
    sdBiasedCoinFlipDiff_1=np.std(biasedCoinFlipDiff_1,axis=0)
    sdBiasedCoinFlipDiff_2=np.std(biasedCoinFlipDiff_2,axis=0)
    seBiasedCoinFlipDiff_1=sdBiasedCoinFlipDiff_1/np.sqrt(num1)
    seBiasedCoinFlipDiff_2=sdBiasedCoinFlipDiff_2/np.sqrt(num2)
    
    # +- SE lines Coinflip
    negDiffsd1=avgBiasedCoinFlipDiff_1-sdBiasedCoinFlipDiff_1
    posDiffsd1=avgBiasedCoinFlipDiff_1+sdBiasedCoinFlipDiff_1
    negDiffsd2=avgBiasedCoinFlipDiff_2-sdBiasedCoinFlipDiff_2
    posDiffsd2=avgBiasedCoinFlipDiff_2+sdBiasedCoinFlipDiff_2
    
    # +- SD lines Coinflip
    negDiffse1=avgBiasedCoinFlipDiff_1-seBiasedCoinFlipDiff_1
    posDiffse1=avgBiasedCoinFlipDiff_1+seBiasedCoinFlipDiff_1
    negDiffse2=avgBiasedCoinFlipDiff_2-seBiasedCoinFlipDiff_2
    posDiffse2=avgBiasedCoinFlipDiff_2+seBiasedCoinFlipDiff_2
    
    avgCumProb_rand1=np.mean(cumProbS_rand1,axis=0)
    avgCumProb_rand2=np.mean(cumProbS_rand2,axis=0)
    sdCumProb_rand1=np.std(cumProbS_rand1,axis=0)
    sdCumProb_rand2=np.std(cumProbS_rand2,axis=0)
    seCumProb_rand1=sdCumProb_rand1/np.sqrt(num1)
    seCumProb_rand2=sdCumProb_rand2/np.sqrt(num2)
    
    # +- SD lines randomised 1 and 2
    pos_randsd1=avgCumProb_rand1+sdCumProb_rand1
    neg_randsd1=avgCumProb_rand1-sdCumProb_rand1
    pos_randsd2=avgCumProb_rand2+sdCumProb_rand2
    neg_randsd2=avgCumProb_rand2-sdCumProb_rand2
    
    # +- SE lines randomised 1 and 2
    pos_randse1=avgCumProb_rand1+seCumProb_rand1
    neg_randse1=avgCumProb_rand1-seCumProb_rand1
    pos_randse2=avgCumProb_rand2+seCumProb_rand2
    neg_randse2=avgCumProb_rand2-seCumProb_rand2
    
    avgCumProb_1=np.mean(cumProbS_1,axis=0)
    avgCumProb_2=np.mean(cumProbS_2,axis=0)
    sdCumProb_1=np.std(cumProbS_1,axis=0)
    sdCumProb_2=np.std(cumProbS_2,axis=0)
    seCumProb_1=sdCumProb_1/np.sqrt(num1)
    seCumProb_2=sdCumProb_2/np.sqrt(num2)
    
    # +- SE lines for 1 and 2
    posse1=avgCumProb_1+seCumProb_1
    negse1=avgCumProb_1-seCumProb_1
    posse2=avgCumProb_2+seCumProb_2
    negse2=avgCumProb_2-seCumProb_2
    
    # +- SD lines for 1 and 2
    possd1=avgCumProb_1+sdCumProb_1
    negsd1=avgCumProb_1-sdCumProb_1
    possd2=avgCumProb_2+sdCumProb_2
    negsd2=avgCumProb_2-sdCumProb_2
    
    x1=range(0,len(avgCumProb_rand1))
    x2=range(0,len(avgCumProb_rand2))
    if np.max(x1)>np.max(x2):
        x=x1
    else: x=x2
    
    ## Plot residual average cumulative probability and SD
    ## this plot shows the residual of the two groups we are comparing and the coinflip
    figName='streakLengthCumProb_DIFF' + name1 + '_vs_' + name2 + '_SD'
    plt.figure(figName,constrained_layout=True)
    
    plt.plot(avgBiasedCoinFlipDiff_1,label=label1,color=col1,linewidth=2)
    plt.plot(negDiffsd1,color=col1,linewidth=0.5,alpha=0.7)
    plt.plot(posDiffsd1,color=col1,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negDiffsd1,posDiffsd1,color=col1,alpha=0.2)
    
    plt.plot(avgBiasedCoinFlipDiff_2,label=label2,color=col2,linewidth=2)
    plt.plot(negDiffsd2,color=col2,linewidth=0.5,alpha=0.7)
    plt.plot(posDiffsd2,color=col2,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negDiffsd2,posDiffsd2,color=col2,alpha=0.2)
    
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.1,0.2,0.3)
    plt.yticks(yint,yint,fontsize=18)
    
    plt.xlabel('Streak Length (bouts)',fontsize=22,labelpad=8)
    plt.ylabel('Residual probability',fontsize=22,labelpad=8)
    plt.legend(fontsize=18,handlelength=1,framealpha=0)
    
    
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()
        
        ## Plot residual average cumulative probability and SE
    ## this plot shows the residual of the two groups we are comparing and the coinflip
    figName='streakLengthCumProb_DIFF' + name1 + '_vs_' + name2 + '_SE'
    plt.figure(figName,constrained_layout=True)
    
    plt.plot(avgBiasedCoinFlipDiff_1,label=label1,color=col1,linewidth=2)
    plt.plot(negDiffse1,color=col1,linewidth=0.5,alpha=0.7)
    plt.plot(posDiffse1,color=col1,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negDiffse1,posDiffse1,color=col1,alpha=0.2)
    
    plt.plot(avgBiasedCoinFlipDiff_2,label=label2,color=col2,linewidth=2)
    plt.plot(negDiffse2,color=col2,linewidth=0.5,alpha=0.7)
    plt.plot(posDiffse2,color=col2,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negDiffse2,posDiffse2,color=col2,alpha=0.2)
    
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.1,0.2,0.3)
    plt.yticks(yint,yint,fontsize=18)
    
    plt.xlabel('Streak Length (bouts)',fontsize=22,labelpad=8)
    plt.ylabel('Residual probability',fontsize=22,labelpad=8)
    plt.legend(fontsize=18,handlelength=1,framealpha=0)
    
    
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()
    
    ## Plot group average cumulative probability and SE
    ## The curve represents the 'streakiness' of the group or individual fish, displayed as the pcumulative probabiity of a random bout belonging to a streak of length x
    figName='streakLengthCumProb_' + name1 + '_vs_' + name2 + '_SE'
    plt.figure(figName,constrained_layout=True)
    
    plt.plot(avgCumProb_rand1,label=labelRand,color='black',linewidth=2)
    plt.plot(neg_randse1,color='black',linewidth=0.5,alpha=0.35)
    plt.plot(pos_randse1,color='black',linewidth=0.5,alpha=0.35)
    plt.fill_between(x1,neg_randse1,pos_randse1,color='black',alpha=0.1)
    
    plt.plot(avgCumProb_rand2,color='black',linewidth=2)
    plt.plot(neg_randse2,color='black',linewidth=0.5,alpha=0.35)
    plt.plot(pos_randse2,color='black',linewidth=0.5,alpha=0.35)
    plt.fill_between(x2,neg_randse2,pos_randse2,color='black',alpha=0.1)
    
    ##
    
#    plt.plot(avgCumProb_coin,label=labelCoin,color='#0f1b8a',linewidth=2)
#    plt.plot(neg_coin,color='#0f1b8a',linewidth=0.5,alpha=0.7)
#    plt.plot(pos_coin,color='#0f1b8a',linewidth=0.5,alpha=0.7)
#    plt.fill_between(x2,neg_coin,pos_coin,color='#0f1b8a',alpha=0.2)
    
    plt.plot(avgCumProb_1,label=label1,color=col1,linewidth=2)
    plt.plot(negse1,color=col1,linewidth=0.5,alpha=0.7)
    plt.plot(posse1,color=col1,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negse1,posse1,color=col1,alpha=0.2)
    
    plt.plot(avgCumProb_2,label=label2,color=col2,linewidth=2)
    plt.plot(negse2,color=col2,linewidth=0.5,alpha=0.7)
    plt.plot(posse2,color=col2,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negse2,posse2,color=col2,alpha=0.2)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.2,0.4,0.6,0.8,1.0)
    plt.yticks(yint,yint,fontsize=18)
    plt.xlabel('Streak Length',fontsize=22)
    plt.ylabel('Cumulative Probability',fontsize=22)
#    plt.title('Cumulative histogram of streak lengths')
    plt.legend(fontsize=18,framealpha=0)
    
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()
        
    ## Plot group average cumulative probability and SD
    ## The curve represents the 'streakiness' of the group or individual fish, displayed as the pcumulative probabiity of a random bout belonging to a streak of length x
    figName='streakLengthCumProb_' + name1 + '_vs_' + name2 + '_SD'
    plt.figure(figName,constrained_layout=True)
    
    plt.plot(avgCumProb_rand1,label=labelRand,color='black',linewidth=2)
    plt.plot(neg_randsd1,color='black',linewidth=0.5,alpha=0.35)
    plt.plot(pos_randsd1,color='black',linewidth=0.5,alpha=0.35)
    plt.fill_between(x1,neg_randsd1,pos_randsd1,color='black',alpha=0.1)
    
    plt.plot(avgCumProb_rand2,color='black',linewidth=2)
    plt.plot(neg_randsd2,color='black',linewidth=0.5,alpha=0.35)
    plt.plot(pos_randsd2,color='black',linewidth=0.5,alpha=0.35)
    plt.fill_between(x2,neg_randsd2,pos_randsd2,color='black',alpha=0.1)
    
    ##
    
#    plt.plot(avgCumProb_coin,label=labelCoin,color='#0f1b8a',linewidth=2)
#    plt.plot(neg_coin,color='#0f1b8a',linewidth=0.5,alpha=0.7)
#    plt.plot(pos_coin,color='#0f1b8a',linewidth=0.5,alpha=0.7)
#    plt.fill_between(x2,neg_coin,pos_coin,color='#0f1b8a',alpha=0.2)
    
    plt.plot(avgCumProb_1,label=label1,color=col1,linewidth=2)
    plt.plot(negsd1,color=col1,linewidth=0.5,alpha=0.7)
    plt.plot(possd1,color=col1,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negsd1,possd1,color=col1,alpha=0.2)
    
    plt.plot(avgCumProb_2,label=label2,color=col2,linewidth=2)
    plt.plot(negsd2,color=col2,linewidth=0.5,alpha=0.7)
    plt.plot(possd2,color=col2,linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negsd2,possd2,color=col2,alpha=0.2)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.2,0.4,0.6,0.8,1.0)
    plt.yticks(yint,yint,fontsize=18)
    plt.xlabel('Streak Length',fontsize=22)
    plt.ylabel('Cumulative Probability',fontsize=22)
#    plt.title('Cumulative histogram of streak lengths')
    plt.legend(fontsize=18,framealpha=0)
    
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()
    
    ## Plot individual curves for 1 with mean on top
    figName='AllStreakCurves_' + name1
    plt.figure(figName,constrained_layout=True)
    for curve in cumProbS_1:
        plt.plot(curve,label='',color=col1,linewidth=2,alpha=0.5)
    plt.plot(avgCumProb_1,label=label1,color=col1,linewidth=2.5)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.2,0.4,0.6,0.8,1.0)
    plt.yticks(yint,yint,fontsize=18)
    plt.xlabel('Streak Length',fontsize=22)
    plt.ylabel('Cumulative Probability',fontsize=22)
    plt.legend(fontsize=18,framealpha=0)
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()
        
    ## Plot individual curves for 2 with mean on top
    figName='AllStreakCurves_' + name2
    plt.figure(figName,constrained_layout=True)
    for curve in cumProbS_2:
        plt.plot(curve,label='',color=col2,linewidth=2,alpha=0.5)
    plt.plot(avgCumProb_2,label=label2,color=col2,linewidth=2.5)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    xint=(0,5,10,15)
    plt.xticks(xint,xint,fontsize=18)
    yint=(0,0.2,0.4,0.6,0.8,1.0)
    plt.yticks(yint,yint,fontsize=18)
    plt.xlabel('Streak Length',fontsize=22)
    plt.ylabel('Cumulative Probability',fontsize=22)
    plt.legend(fontsize=18,framealpha=0)
    if saveFigures:
        AZU.cycleMkDir(savepath)
        saveName=savepath+'\\'+figName+'.png'
        plt.savefig(saveName,dpi=600)
        
    if keepFigures==False:
        plt.close()

    cumProbse_1=posse1-avgCumProb_1
    cumProbse_2=posse2-avgCumProb_2
    return avgCumProb_1,cumProbse_1,avgCumProb_2,cumProbse_2,avgCumProb_rand1
 
def FLRBoutCompare(dic1,dic2,savepath=r'D:\\Shelf\\',plotInd=True,keepFigures=False, saveFigures=True,label1='NO LABEL!',label2='NO LABEL!',col1='#486AC6',col2='#F3930C'):
    
    name1=dic1['Name']
    name2=dic2['Name']
    
    Dic1SeqProbs1=dic1['avgData']['avg_seqProbS1']
    Dic1SeqProbs2=dic1['avgData']['avg_seqProbS2_Z']
    Dic2SeqProbs1=dic2['avgData']['avg_seqProbS1']
    Dic2SeqProbs2=dic2['avgData']['avg_seqProbS2_Z']
    
    # first order measures
    Dic1SeqProbs1_AV=dic1['Metrics']['seqProbs1']['Mean']
    Dic1SeqProbs1_SE=dic1['Metrics']['seqProbs1']['SEM']
    Dic2SeqProbs1_AV=dic2['Metrics']['seqProbs1']['Mean']
    Dic2SeqProbs1_SE=dic2['Metrics']['seqProbs1']['SEM']
    
    # second order measures
    Dic1SeqProbs2_AV=dic1['Metrics']['seqProbs2_Z']['Mean']
    Dic1SeqProbs2_SE=dic1['Metrics']['seqProbs2_Z']['SEM']
    Dic2SeqProbs2_AV=dic2['Metrics']['seqProbs2_Z']['Mean']
    Dic2SeqProbs2_SE=dic2['Metrics']['seqProbs2_Z']['SEM']
    
    comb1=dic1['Metrics']['seqProbs1']['comb']
    comb2=dic1['Metrics']['seqProbs2_Z']['comb']
    
    numFish1=len(Dic1SeqProbs1)
    numFish2=len(Dic2SeqProbs1)
    n_groups1=len(comb1)
    n_groups2=len(comb2)
    
    fig, ax = plt.subplots()
    index = np.arange(n_groups1)
    bar_width = 0.35
    opacity = 0.8
    
    # plot the average and se for Dic1
    plt.bar(index, list(Dic1SeqProbs1_AV), bar_width,
    alpha=opacity,
    color=col1,
    label=label1,
    yerr=list(Dic1SeqProbs1_SE))
    
    # plot individual points maybe
    if plotInd:
        for j in range(numFish1):
            plt.scatter(index,Dic1SeqProbs1[j],alpha=opacity,color='black',s=8)
    
    # plot the average and se for Dic2     
    plt.bar(index + bar_width+.1, Dic2SeqProbs1_AV, bar_width,
    alpha=opacity,
    color=col2,
    label=label2,
    yerr=list(Dic2SeqProbs1_SE))
    
    # plot individual points maybe
    if plotInd:
        for j in range(numFish2):
            plt.scatter(index + bar_width+.1,Dic2SeqProbs1[j],alpha=opacity,color='black',s=8)
    
    plt.xlabel('Bout direction',fontsize=22,labelpad=8)
    plt.ylabel('Relative Frequency',fontsize=22,labelpad=8)
#    plt.title('Proportion of bout directions')
    plt.xticks(index + (bar_width*0.5)+.1, tuple(comb1),fontsize=18)
    yint=(0,0.2,0.4,0.6)
    plt.yticks(yint,yint,fontsize=18)
    plt.legend(fontsize=18,handlelength=1,framealpha=0)
    
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    
    plt.tight_layout()
    plt.show()
    saveName=savepath + r'\\Comparison_' + name1 + '_vs_' + name2 + '_BoutProps.png'
    if saveFigures:
        AZU.cycleMkDir(savepath)
        plt.savefig(saveName,dpi=600)
    if keepFigures==False:plt.close()
    #######################
    ### Second Order bout sequencing
    # Sort by most overrepresented to most underrepresented in Ctrl fish (dic1)
    arr1inds = np.asarray(Dic1SeqProbs2_AV.argsort())
    sorted_Dic1SeqProbs2_AV = np.asarray(Dic1SeqProbs2_AV[arr1inds[::-1]])
    sorted_Dic1SeqProbs2_SE = np.asarray(Dic1SeqProbs2_SE[arr1inds[::-1]])
    sorted_Dic2SeqProbs2_AV = np.asarray(Dic2SeqProbs2_AV[arr1inds[::-1]])
    sorted_Dic2SeqProbs2_SE = np.asarray(Dic2SeqProbs2_SE[arr1inds[::-1]])
    sorted_comb2 = [comb2[i] for i in list(arr1inds[::-1])]
    
    fig, ax = plt.subplots()
    index = np.arange(n_groups2)
    bar_width = 0.35
    opacity = 0.8
    
    plt.bar(index, sorted_Dic1SeqProbs2_AV, bar_width,
    alpha=opacity,
    color=col1,
    label=label1,
    yerr=sorted_Dic1SeqProbs2_SE)
    if plotInd:
        for j in range(numFish1):
            sorted_d1s2 = np.asarray(Dic1SeqProbs2[j])[arr1inds[::-1]]
            plt.scatter(index,sorted_d1s2,alpha=opacity,color='black',s=6)
            
    plt.bar(index + bar_width, sorted_Dic2SeqProbs2_AV, bar_width,
    alpha=opacity,
    color=col2,
    label=label2,
    yerr=sorted_Dic2SeqProbs2_SE)
    
    for j in range(numFish2):
            sorted_d2s2 = np.asarray(Dic2SeqProbs2[j])[arr1inds[::-1]]
            plt.scatter(index + bar_width,sorted_d2s2,alpha=opacity,color='black',s=6)
    
    plt.xlabel('Bout pair',fontsize=22,labelpad=8)
    plt.ylabel('Z-score',fontsize=22,labelpad=8)        
#    plt.title('Second order sequencing of bouts')
    plt.xticks(index + (bar_width*0.5), tuple(sorted_comb2),fontsize=18)
    yint=(-15,-10,-5,0,5,10,15,20,25)
    plt.yticks(yint,yint,fontsize=18)
    plt.legend(fontsize=18,handlelength=1,framealpha=0)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    plt.tight_layout()
    plt.show()
    saveName=savepath + r'\\Comparison_' + name1 + '_vs_' + name2 + '_BoutPairs.png'
    if saveFigures==1:
        AZU.cycleMkDir(savepath)
        plt.savefig(saveName,dpi=600)
    if keepFigures==False:plt.close()
    
#n_groups = len(comb2)
#mean1=probs2AVG_WT
#se1=probs2SE_WT
#mean2=avg_seqProbS2
#se2=se_seqProbS2
#
## create plot


### makes three ROI figures: each 3 x 3 according to which ROI we are looking at. 
    
  ## 1. Aspirated vs control raw proportions of individual bout types
  ## 2. Aspirated vs control raw probabilities of bout pairs
  ## 3. Aspirated vs control zscores from 10k randomly shuffled sequences showing overrepresentation of specific bout pairs in each ROI
  
def ROISeqFromDict(CtrlDic,AspDic,saveFigures=1,keepFigures=1,savepath=r'D:\\ROI_Figures\\'):
    
    AZU.cycleMkDir(savepath)
    label1='Control'
    label2='Aspirated'
    col1='green'
    col2='magenta'
    
    CtrlROIs = CtrlDic['ROIs']
    AspROIs  = AspDic['ROIs']
    
    comb1=CtrlROIs[0]['seqProbs1']['comb']
    comb2=CtrlROIs[0]['seqProbs2']['comb']
    n_groups1=len(comb1)
    n_groups2=len(comb2)
    
    # check the order of overrepesentation overall in Ctrl and stick to this ordering throughout ROIs
#    arrinds = np.asarray(CtrlDic['Metrics']['seqProbs1']['Mean'].argsort())
#    arr1inds = np.asarray(CtrlDic['Metrics']['seqProbs2_Z']['Mean'].argsort())
#    arrinds = np.asarray(np.nanmean(AspROIs[5]['seqProbs1']['prob'],axis=0).argsort())
#    arr1inds = np.asarray(np.nanmean(AspROIs[5]['seqProbs2']['prob'],axis=0).argsort())
    arr1inds=np.asarray([0,4,8,1,2,3,6,5,7])
    arrinds=np.asarray([0,1,2])
    comb1 = [comb1[i] for i in list(arrinds)]
    comb2 = [comb2[i] for i in list(arr1inds)]
    
    ylim1=(0,0.7)
    ylim2=(0,0.5)
    ylimz=(-30,100)
    for i in range(len(CtrlROIs)):
        ROIName=AspROIs[i]['ROIName']
        ## sort all values for this ROI in the same way
#        a1=AspROIs[i]['seqProbs1']['prob']
#        nn=len(a1)
#        a1[a1==-1]=np.nan
#        nnn=a1[a1!=np.nan]
#        n=len(nnn)/nn
#        a1m_unsort=np.nanmean(a1,axis=0)
#        a1m=a1m_unsort[arrinds]
#        a1_err_unsort = np.nanstd(a1,axis=0)/np.sqrt(n)
#        a1_err=a1_err_unsort[arrinds]
#        
#        c1=CtrlROIs[i]['seqProbs1']['prob']
#        nn=len(c1)
#        c1[c1==-1]=np.nan
#        nnn=c1[c1!=np.nan]
#        n=len(nnn)/nn
#        c1m_unsort=np.nanmean(c1,axis=0)
#        c1m=c1m_unsort[arrinds]
#        c1_err_unsort = np.nanstd(c1,axis=0)/np.sqrt(n)
#        c1_err=c1_err_unsort[arrinds]
#        
#        
#        c2=CtrlROIs[i]['seqProbs2']['prob']
#        nn=len(c2)
#        c2[c2==-1]=np.nan
#        nnn=c2[c2!=np.nan]
#        n=len(nnn)/nn
#        c2m_unsort=np.nanmean(c2,axis=0)
#        c2m=c2m_unsort[arr1inds]
#        c2_err_unsort = np.nanstd(c2,axis=0)/np.sqrt(n)
#        c2_err=c2_err_unsort[arr1inds]
#        
#        a2=AspROIs[i]['seqProbs2']['prob']
#        nn=len(a2)
#        a2[a2==-1]=np.nan
#        nnn=a2[a2!=np.nan]
#        n=len(nnn)/nn
#        a2m_unsort=np.nanmean(a2,axis=0)
#        a2m=a2m_unsort[arr1inds]
#        a2_err_unsort = np.nanstd(a2,axis=0)/np.sqrt(n)
#        a2_err=a2_err_unsort[arr1inds]
#        
        
        az=AspROIs[i]['seqProbs2']['zscores']
        nn=len(az)
        az[az==-1]=np.nan
        az[az==30]=np.nan
        az[np.isfinite(az)!=True]=np.nan
        nnn=az[az!=np.nan]
        n=len(nnn)/nn
        azm_unsort=np.nanmean(az,axis=0)
        azm=azm_unsort[arr1inds[::-1]]
        az_err_unsort = np.nanstd(az,axis=0)/np.sqrt(n)
        az_err=az_err_unsort[arr1inds[::-1]]
        
        cz=CtrlROIs[i]['seqProbs2']['zscores']
        nn=len(cz)
        cz[cz==-1]=np.nan
        cz[cz==30]=np.nan
        cz[np.isfinite(cz)!=True]=np.nan
        nnn=cz[cz!=np.nan]
        n=len(nnn)/nn
        czm_unsort=np.nanmean(cz,axis=0)
        czm=czm_unsort[arr1inds[::-1]]
        cz_err_unsort = np.nanstd(cz,axis=0)/np.sqrt(n)
        cz_err=cz_err_unsort[arr1inds[::-1]]
        
#        figName = 'BoutFreq_ROI' + str(i) + '_' + ROIName
#        makeFig(c1m,c1_err,a1m,a1_err,comb1,figName,i,ylim=ylim1)
#        if saveFigures==1:saveAsUsual(savepath,figName)
#        if keepFigures!=1:plt.close()
#
#        figName = 'BoutPairFreq_ROI' + str(i) + '_' + ROIName
#        makeFig(c2m,c2_err,a2m,a2_err,comb2,figName,i,ylim=ylim2)
#        if saveFigures==1:saveAsUsual(savepath,figName)
#        if keepFigures!=1:plt.close()
        
        figName = 'BoutPairZScore_ROI' + str(i) + '_' + ROIName
        makeFig(czm,cz_err,azm,az_err,comb2,figName,i,ylim=ylimz)
        if saveFigures==1:saveAsUsual(savepath,figName)
        if keepFigures!=1:plt.close()
#        
#        
#        RTurns=thisROI['BoutSeq']['Right']
#        FSwims=thisROI['BoutSeq']['Forward']
#                
#        if thisROI['ROIName']=='Top Left':
#            loc=(0,0)
#        if thisROI['ROIName']=='Top':
#            loc=(0,1)
#        if thisROI['ROIName']=='Top Right':
#            loc=(0,2)
#            ax[loc].legend(handles=[FSwimax, LTurnax,RTurnax],loc='upper right')
#        if thisROI['ROIName']=='Middle Left':
#            loc=(1,0)
#        if thisROI['ROIName']=='Central Chamber':
#            loc=(1,1)
#        if thisROI['ROIName']=='Middle Right':
#            loc=(1,2)
#        if thisROI['ROIName']=='Bottom Left':
#            loc=(2,0)
#        if thisROI['ROIName']=='Bottom':
#            loc=(2,1)
#        if thisROI['ROIName']=='Bottom Right':
#            loc=(2,2)
            
#            
#        FSwimax=ax[loc].scatter(range(1,len(FSwims)+1),FSwims,color='gray',s=100,label='Forward Swims')
#        LTurnax=ax[loc].scatter(range(1,len(LTurns)+1),LTurns+0.1,color='b',s=100,label='Left Turns')
#        RTurnax=ax[loc].scatter(range(1,len(RTurns)+1),RTurns-0.1,color='r',s=100,label='Right Turns')
#        ax[loc].set_ylim(0.4,2)
##        plt.xlim(0,100)
#        ax[loc].set_yticklabels(labels=['Right','Forward','Left'])
#        
#        
#    figname='Turn Sequences by ROI'
#    fig.show()
        

def saveAsUsual(savepath,figName):
    saveName=savepath + r'\\' + figName + '.png'
    AZU.cycleMkDir(savepath)
    plt.savefig(saveName,dpi=600)

def makeFig(c,cerr,a,aerr,comb,figName,i,col1='green',col2='magenta',label1='Ctrl',label2='Asp',ylim=(0,0.7)):
    
    ng=len(comb)
    ## Figure 1 ## First Order
    fig, ax = plt.subplots()
    index = np.arange(ng)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, a, bar_width,
    alpha=opacity,
    color=col2,
    label=label2,
    yerr=aerr)

    plt.bar((index + bar_width)+4, c, bar_width,
            alpha=opacity,
            color=col1,
            label=label1,
            yerr=cerr)

    plt.xlabel('Bout type')
    plt.ylabel('Relative Frequency')
    plt.ylim(ylim)
    plt.title('Proportion of bout type frequency' + ' ROI ' + str(i))
    plt.xticks(index + (bar_width*0.5)+4, tuple(comb),fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()
            
def ROIGroupFigs(GroupedFish,ROI_BPS,ROI_avgVelocity,ROI_avgAngVelocityBout,ROI_biasLeftBout,ROI_LTurnPC,ROI_AvgBoutAmps,ROI_AvgBoutDists,ROI_PCTimeSpent,ROIMasks,name,ROINames,savepath,FPS=120,save=True,keep=False):
   
    # Grab overall numbers as well as ROI numbers from dictionary
    avg=GroupedFish['avgData']
    avgBPSs=avg['BPSs']
    avgVel=avg['avgVelocity']
    avgAngVel=avg['avgAngVelocityBout']
    avgBiasLeftBout=avg['biasLeftBout']
    avgLTurnPC=avg['LTurnPC']                
    
    
    ############### BPS
    ROI_BPS_nans=np.copy(ROI_BPS)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_BPS_nans[ROI_BPS==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_BPS_nans)
    filtered_data = [d[m] for d, m in zip(ROI_BPS_nans.T, mask.T)]
    filtered_data.append(avgBPSs) # add info for all ROIS
    saveName=-1
    figname='avgBPS Comparison. ROIs'
    figsize=[10,6]
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    labels=ROINames
    labels.append('All ROIs')
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Bouts per second')
    plt.xlabel('ROI Position')

    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_BPS_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # Average Velocity
    ROI_avgVelocity_nans=np.copy(ROI_avgVelocity)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_avgVelocity_nans[ROI_avgVelocity==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_avgVelocity_nans)
    filtered_data = [d[m] for d, m in zip(ROI_avgVelocity_nans.T, mask.T)]
    filtered_data.append(avgVel) # add info for overall avg
    saveName=-1
    figname='Velocity Comparison. ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Velocity (mm/s)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_avgVel_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # Average Angular Velocity
    ROI_avgAngVelocity_nans=np.copy(ROI_avgAngVelocityBout)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_avgAngVelocity_nans[ROI_avgAngVelocityBout==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_avgAngVelocity_nans)
    filtered_data = [d[m] for d, m in zip(ROI_avgAngVelocity_nans.T, mask.T)]
    filtered_data.append(avgAngVel) # add info for all ROIS
    saveName=-1
    figname='Angular Velocity Comparison. ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Velocity (mm/s)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_avgAngVel_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # ROI_biasLeftBout
    ROI_biasLeftBout_nans=np.copy(ROI_biasLeftBout)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_biasLeftBout_nans[ROI_biasLeftBout==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_biasLeftBout_nans)
    filtered_data = [d[m] for d, m in zip(ROI_biasLeftBout_nans.T, mask.T)]
    filtered_data.append(avgBiasLeftBout) # add info for all ROIS
    saveName=-1
    figname='Left bout bias (angle) Comparison ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Overall Leftward Bias (radians)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_biasLeftBout_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # ROI_LTurnPC
    ROI_LTurnPC_nans=np.copy(ROI_LTurnPC)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_LTurnPC_nans[ROI_LTurnPC==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_LTurnPC_nans)
    filtered_data = [d[m] for d, m in zip(ROI_LTurnPC_nans.T, mask.T)]
    filtered_data.append(avgLTurnPC) # add info for all ROIS
    saveName=-1
    figname='Left turn % Comparison ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Left Turns (% of turning bouts >5 deg)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_LTurnPC_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
      
    # ROI_AvgBoutAmps
    ROI_AvgBoutAmps_nans=np.copy(ROI_AvgBoutAmps)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_AvgBoutAmps_nans[ROI_AvgBoutAmps==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_AvgBoutAmps_nans)
    filtered_data = [d[m] for d, m in zip(ROI_AvgBoutAmps_nans.T, mask.T)]
#    filtered_data.append(avgAmps) # add info for all ROIS
    saveName=-1
    figname='Avg bout amplitude Comparison: ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylabel('Peak bout energy (AU)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_boutAmps_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # ROI_AvgBoutDists
    ROI_AvgBoutDists_nans=np.copy(ROI_AvgBoutDists)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_AvgBoutDists_nans[ROI_AvgBoutDists==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_AvgBoutDists_nans)
    filtered_data = [d[m] for d, m in zip(ROI_AvgBoutDists_nans.T, mask.T)]
#    filtered_data.append(avgDists) # add info for all ROIS
    saveName=-1
    figname='Avg bout distance Comparison: ROIs'
    plt.figure(figname,figsize=figsize)
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylim((0,0.6))
    plt.ylabel('avg Distance per Bout (cm)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_boutDists_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # ROI_AvgBoutDists
    ROI_PCTimeSpent_nans=np.copy(ROI_PCTimeSpent)
    
    # Filter data for missing datapoints (empty ROIS) using np.isnan
    ROI_PCTimeSpent_nans[ROI_PCTimeSpent==-1]=np.nan # remove ROIs for which we have no info
    mask = ~np.isnan(ROI_PCTimeSpent_nans)
    filtered_data = [d[m] for d, m in zip(ROI_PCTimeSpent_nans.T, mask.T)]
    saveName=-1
    figname='Proportion time spent Comparison: ROIs'
    plt.figure(figname,figsize=[9,6])
    plt.boxplot(filtered_data,notch=False,showfliers=False)
    plt.title(figname)
    ticks,_=plt.xticks()
    plt.xticks(ticks,labels,rotation=20)
    plt.ylim(0,0.6)
    plt.ylabel('Proportion of time spent (%)')
    plt.xlabel('ROI Position')
    
    for i in ticks:
        y = filtered_data[i-1]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i, 0.08, size=len(y))
        plt.plot(x, y, 'b.', alpha=0.5)
        
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_TimeSpent_ROIs.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
###############################################################################
def compareGroupStats3Dics(dic1File,dic2File,dic3File,FPS=120,save=True,keep=False):
        
    dic1Name,avgCumDistAV_1,avgCumDistSEM_1,avgBoutAmps_1,allBPS_1,avgBoutAV_1,avgBoutSEM_1,avgHeatmap_1,avgVelocity_1=AZA.unpackGroupDictFile(dic1File)
    dic2Name,avgCumDistAV_2,avgCumDistSEM_2,avgBoutAmps_2,allBPS_2,avgBoutAV_2,avgBoutSEM_2,avgHeatmap_2,avgVelocity_2=AZA.unpackGroupDictFile(dic2File)
    dic3Name,avgCumDistAV_3,avgCumDistSEM_3,avgBoutAmps_3,allBPS_3,avgBoutAV_3,avgBoutSEM_3,avgHeatmap_3,avgVelocity_3=AZA.unpackGroupDictFile(dic3File)
#    sF=0
#    eF=36000
#    pstr='0-5min'
#    
#    avgCumDistAV_1
#    avgCumDistSEM_1
#    avgCumDistAV_2
#    avgCumDistSEM_2
#    avgBoutAmps_1
#    avgBoutAmps_2
#    allBPS_1
    ################### avgBout
    saveName=-1
    compName=dic1Name + ' vs ' + dic2Name + 'vs' + dic3Name
    figname='avgBout Comparison. Groups:'+ compName
    plt.figure(figname)
    
    xFr=range(len(avgBoutAV_1))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_1)
    pos1=avgBoutAV_1+avgBoutSEM_1
    neg1=avgBoutAV_1-avgBoutSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgBoutAV_2))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_2)
    pos2=avgBoutAV_2+avgBoutSEM_2
    neg2=avgBoutAV_2-avgBoutSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgBoutAV_3))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_3)
    pos3=avgBoutAV_3+avgBoutSEM_3
    neg3=avgBoutAV_3-avgBoutSEM_3
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### cumDist
    saveName=-1
    xFr=range(len(avgCumDistAV_1))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1)
    pos1=avgCumDistAV_1+avgCumDistSEM_1
    neg1=avgCumDistAV_1-avgCumDistSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2)
    pos2=avgCumDistAV_2+avgCumDistSEM_2
    neg2=avgCumDistAV_2-avgCumDistSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3)
    pos3=avgCumDistAV_3+avgCumDistSEM_3
    neg3=avgCumDistAV_3-avgCumDistSEM_3
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    # Chi-square Test
    shortest=np.min([len(avgCumDistAV_1),len(avgCumDistAV_2)])
    chisq,pvalue=stats.chisquare(avgCumDistAV_1[0:shortest], f_exp=avgCumDistAV_2[0:shortest])
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDist.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ################## cumDist Zoom 15min
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:(63600*2)]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:(63600*2)]
    avgCumDistAV_3Zoom=avgCumDistAV_3[0:(63600*2)]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:(63600*2)]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:(63600*2)]
    avgCumDistSEM_3Zoom=avgCumDistSEM_3[0:(63600*2)]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 15 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3Zoom)
    pos3=avgCumDistAV_3Zoom+avgCumDistSEM_3Zoom
    neg3=avgCumDistAV_3Zoom-avgCumDistSEM_3Zoom
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom15min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### boutAmps
    dBA = [avgBoutAmps_1, avgBoutAmps_2,avgBoutAmps_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgBoutAmps Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Velocity (mm/frame)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgBoutAmps_2, avgBoutAmps_3, axis=0, equal_var=False)
    plt.legend(['p 2 vs 3 = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBoutAmps.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()

    ################## cumDist Zoom 5min
    ll=5*60*FPS
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:ll]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:ll]
    avgCumDistAV_3Zoom=avgCumDistAV_3[0:ll]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:ll]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:ll]
    avgCumDistSEM_3Zoom=avgCumDistSEM_3[0:ll]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 5 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3Zoom)
    pos3=avgCumDistAV_3Zoom+avgCumDistSEM_3Zoom
    neg3=avgCumDistAV_3Zoom-avgCumDistSEM_3Zoom
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
#    # Fisher's Exact Test
#    rpy2.robjects.numpy2ri.activate()
#    m = np.array([avgCumDistAV_1Zoom,avgCumDistAV_2Zoom])
#    res = stats.fisher_test(m)
#    pvalue=res[0][0]
#    plt.legend(['p = ' + str(round(pvalue,3))])
#    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom5min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()    
    
    #################### avgVel
    dBA = [avgVelocity_1, avgVelocity_2,avgVelocity_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgVelocity Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Velocity (mm/sec)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgVelocity_2, avgVelocity_3, axis=0, equal_var=False)
    plt.legend(['p 2 vs 3 = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgVelocity.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### heatmapDiff
    
    avgDiffHeatmap=avgHeatmap_1-avgHeatmap_2
    saveName=-1
    figname='Difference between heatmaps. Groups:'+ compName
    plt.figure(figname)
    plt.imshow(avgDiffHeatmap)
    plt.title(figname)
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_diffHeatmap.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ############### BPS
    dBA = [allBPS_1, allBPS_2, allBPS_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgBPS Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Bouts per second')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(allBPS_2, allBPS_3, axis=0, equal_var=False)
    plt.legend(['p = 2 vs 3 ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBPS.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    
    
    
def compareGroupStats(dic1File,dic2File,FPS=120,save=True,keep=False):
        
    dic1Name,avgCumDistAV_1,avgCumDistSEM_1,avgBoutAmps_1,avgBoutDists_1,allBPS_1,avgBoutAV_1,avgBoutSEM_1,avgHeatmap_1,avgVelocity_1=AZA.unpackGroupDictFile(dic1File)
    dic2Name,avgCumDistAV_2,avgCumDistSEM_2,avgBoutAmps_2,avgBoutDists_2,allBPS_2,avgBoutAV_2,avgBoutSEM_2,avgHeatmap_2,avgVelocity_2=AZA.unpackGroupDictFile(dic2File)
#    sF=0
#    eF=36000
#    pstr='0-5min'
#    
#    avgCumDistAV_1
#    avgCumDistSEM_1
#    avgCumDistAV_2
#    avgCumDistSEM_2
#    avgBoutAmps_1
#    avgBoutAmps_2
#    allBPS_1
    ################### avgBout
    saveName=-1
    xFr=range(len(avgBoutAV_1))
    x=np.divide(xFr,FPS)
    compName=dic1Name + ' vs ' + dic2Name
    figname='avgBout Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgBoutAV_1)
    pos1=avgBoutAV_1+avgBoutSEM_1
    neg1=avgBoutAV_1-avgBoutSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    plt.plot(x,avgBoutAV_2)
    pos2=avgBoutAV_2+avgBoutSEM_2
    neg2=avgBoutAV_2-avgBoutSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### cumDist
    saveName=-1
    xFr=range(len(avgCumDistAV_1))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1)
    pos1=avgCumDistAV_1+avgCumDistSEM_1
    neg1=avgCumDistAV_1-avgCumDistSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2)
    pos2=avgCumDistAV_2+avgCumDistSEM_2
    neg2=avgCumDistAV_2-avgCumDistSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    # Chi-square Test
    shortest=np.min([len(avgCumDistAV_1),len(avgCumDistAV_2)])
    chisq,pvalue=stats.chisquare(avgCumDistAV_1[0:shortest], f_exp=avgCumDistAV_2[0:shortest])
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDist.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ################## cumDist Zoom 15min
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:(63600*2)]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:(63600*2)]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:(63600*2)]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:(63600*2)]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 15 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom15min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### boutAmps
    dBA = [avgBoutAmps_1, avgBoutAmps_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgBoutAmps Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Velocity (mm/frame)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBoutAmps.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()

    ################## cumDist Zoom 5min
    ll=5*60*FPS
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:ll]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:ll]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:ll]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:ll]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 5 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
#    # Fisher's Exact Test
#    rpy2.robjects.numpy2ri.activate()
#    m = np.array([avgCumDistAV_1Zoom,avgCumDistAV_2Zoom])
#    res = stats.fisher_test(m)
#    pvalue=res[0][0]
#    plt.legend(['p = ' + str(round(pvalue,3))])
#    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom5min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()    
    
    #################### avgVel
    dBA = [avgVelocity_1, avgVelocity_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgVelocity Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Velocity (mm/sec)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgVelocity.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### heatmapDiff
    
    avgDiffHeatmap=avgHeatmap_1-avgHeatmap_2
    saveName=-1
    figname='Difference between heatmaps. Groups:'+ compName
    plt.figure(figname)
    plt.imshow(avgDiffHeatmap)
    plt.title(figname)
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_diffHeatmap.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ############### BPS
    dBA = [allBPS_1, allBPS_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgBPS Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Bouts per second')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBPS.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()

###############################################################################    
def groupFigs(GDict,name,savepath,FPS=120,save=True,keep=False):
    
    numFish=len(GDict['Ind_fish'])
    met=GDict['Metrics']
    pool=GDict['avgData']
    ############################### CumDist ###################################
    ############################### AllFish ###################################
    meanCumDist=met['cumDist']['Mean']
    xFr=range(len(meanCumDist))
    x=np.divide(xFr,FPS)
    
    plt.figure('All Fish, Cumulative Distance travelled')

    # check individual fish cumulative distances are not longer than the mean (they were not included in the mean anyway)    
    mmL=len(meanCumDist)
    fcds=pool['cumDists']
    for fcd in fcds:
        if(len(fcd)>=mmL):
            plt.plot(x,fcd[0:mmL],alpha=0.5,color='gray',linewidth=0.5)
            
    plt.plot(x,meanCumDist,color='blue',linewidth=2,alpha=1)
    plt.title('Cumulative distances of all fish. Group:'+ name)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_allCumDists.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
        
    ###################### Group Mean + SEM ###################################
    SEMCumDist=met['cumDist']['SEM']
    plt.figure('Pooled cumDist')
    plt.plot(x,meanCumDist)
    pos=meanCumDist+SEMCumDist
    neg=meanCumDist-SEMCumDist
    
    plt.plot(x,neg,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg,pos,alpha=0.2)
    plt.title('Group mean cumulative distance. Group:'+ name)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_groupCumDist.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
        
    ####################### avgBout ##################################
    meanavgBout=met['avgBout']['Mean']
    SEMavgBout=met['avgBout']['SEM']
    xFr=range(len(meanavgBout))
    x=np.divide(xFr,FPS)
    
    plt.figure('All Fish, avgBout')
    plt.plot(x,meanavgBout)
    for i in range(numFish):
        plt.plot(x,pool['avgBouts'][i],alpha=0.5,color='gray',linewidth=0.5)
        
    plt.title('Pooled avgBouts: Group:'+ name)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_pooledAvgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
    
    plt.figure('Group avgBout')
    plt.plot(x,meanavgBout)
    pos=meanavgBout+SEMavgBout
    neg=meanavgBout-SEMavgBout
    
    plt.plot(x,neg,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg,pos,alpha=0.2)
    
    plt.title('Group avgBout: Group:'+ name)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_groupAvgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
    
    ########################## Heatmaps ################################
    heatmapAV=met['avgHeatmap']['Mean']
    plt.figure('Relative time spent heatmap')
    plt.imshow(heatmapAV)
    plt.title('Time spent heat map; Group:' + name)
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_groupHeatmapAV.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
    
    heatmapSEM=met['avgHeatmap']['SEM']
    plt.figure('Variance in time spent heatmap')
    plt.imshow(heatmapSEM)
    plt.title('Variance in time spent heat map; Group:' + name)
    
    if(save):
        saveName=savepath+r'\\'+name+r'\\'+name+'_groupHeatmapVariance.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
    ########################## Histograms for... ###########################
    
    # collect indFishBPS, inFishavgVelocity and indFishboutsAmp
    avgAmps=pool['boutAmps']
    allBPS=pool['BPSs']
    allVel=pool['avgVelocity']
    allAngVel=pool['avgAngVelocityBout']
    allBiasLeftBout=pool['biasLeftBout']
    allLTurnPC=pool['LTurnPC']
    
    
    # avgAmps 20 bins Histogram
    ampHist20bins,c=np.histogram(avgAmps,  bins=20, range=(-.001,7))
    ampcenters20bins = (c[:-1]+c[1:])/2
    
    # avgAmps 40 bins Histogram
    ampHist40bins,c=np.histogram(avgAmps,  bins=40, range=(-.001,7))
    ampcenters40bins = (c[:-1]+c[1:])/2
    
    # BPS Histogram
    bpsHist,c=np.histogram(allBPS,  bins=16, range=(-.2,2))
    bpscenters = (c[:-1]+c[1:])/2
    
    # avgVelocity Histogram
    velHist,c=np.histogram(allVel,  bins=16, range=(0,10))
    velcenters = (c[:-1]+c[1:])/2
    
    # biasLeftBout Histogram
    biasHist,c=np.histogram(allBiasLeftBout,  bins=10, range=(-1,1))
    biascenters = (c[:-1]+c[1:])/2
    
    # LTurnPC Histogram
    LTurnHist,c=np.histogram(allLTurnPC,  bins=32, range=(20,80))
    LTurncenters = (c[:-1]+c[1:])/2
    
    # Angular velocity Histogram
    angVelHist,c=np.histogram(allAngVel,  bins=16, range=(-.001,40))
    angVelcenters = (c[:-1]+c[1:])/2
    
    # plot and save boutAmps 20 bins
    figname='Group Bout Amplitude Distribution (20 bins). Group: ' + name
    plt.figure(figname)
    plt.plot(ampcenters20bins, ampHist20bins, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Bout amplitude (mm/s)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_groupBoutAmpsHist20bins.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # plot and save boutAmps 40 bins
    figname='Group Bout Amplitude Distribution (40 bins). Group: ' + name
    plt.figure(figname)
    plt.plot(ampcenters40bins, ampHist40bins, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Bout amplitude (mm/s)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_groupBoutAmpsHist40bins.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # plot and save BPS
    figname='Group BPS Distribution. Group: ' + name
    plt.figure(figname)
    plt.plot(bpscenters, bpsHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('BPS')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName=savepath + r'\\' + name + r'\\' + name + '_avgBPSHist.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # plot and save avgVelocity
    figname='Average Velocity Distribution. Group: ' + name
    plt.figure(figname)
    plt.plot(velcenters, velHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Average Velocity (mm/s)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_avgVelHist.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
     # plot and save avgAngVelocity
    figname='Average Angular Velocity Distribution. Group: ' + name
    plt.figure(figname)
    plt.plot(angVelcenters, angVelHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Average Angular Velocity (degrees/s)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_avgAngVelHist.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # plot and save biasLeftBout
    figname='Angular bias. Group: ' + name
    plt.figure(figname)
    plt.plot(biascenters, biasHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Leftward Bias')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_LBiasHist.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # plot and save LTurnPC
    figname='Left turn bias. Group: ' + name
    plt.figure(figname)
    plt.plot(LTurncenters, LTurnHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Left turns (%)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_LTurnPC.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    # Can also plot out all the velocities, amps and BPSs simply sorted to understand distribution better
    BPS_s=sorted(allBPS)
    Vel_s=sorted(allVel)
    Amps_s=sorted(avgAmps)
    angvel_s=sorted(allAngVel)
    LTPC_s=sorted(allLTurnPC)
    LBias_s=sorted(allBiasLeftBout)

    ## BPS
    figname='All Fish Sorted Mean BPS, Group:' + name
    plt.figure(figname)
    plt.plot(BPS_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Bouts Per Second')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedBPS.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
         
    # Vels
    figname='All Fish Sorted Mean Velocities, Group:' + name
    plt.figure(figname)
    plt.plot(Vel_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Average Velocity (mm/s)')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedVel.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    

    # Angular vels        
    figname='All Fish Sorted Mean AngVelocities, Group:' + name
    plt.figure(figname)
    plt.plot(angvel_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Average Angular Velocity (degrees/s)')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedAngVel.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # LeftTurnPercent
    figname='All Fish Sorted Left turn percent, Group:' + name
    plt.figure(figname)
    plt.plot(LTPC_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Left Turns (%)')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedLTurnPC.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
        
    # Left bias
    figname='All Fish Sorted Left bias, Group:' + name
    plt.figure(figname)
    plt.plot(LBias_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Left bias index')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedLBias.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    
    # Amps
    figname='All Fish Sorted Mean Bout Amplitudes, Group:' + name
    plt.figure(figname)
    plt.plot(Amps_s)
    plt.xlabel('FishNo.')
    plt.ylabel('Mean Peak Bout Velocity (mm/s)')
    plt.title(figname)
    if(save):
        saveName = savepath + r'\\' + name + r'\\' + name +'_sortedBoutAmps.png'
        AZU.cycleMkDir(savepath+r'\\'+name+r'\\')
        plt.savefig(saveName,dpi=600)
    if keep==False:
        plt.close('all')
    
def indBoutAmpsHistFig(boutAmps,name,savepath,FPS=120,save=True,keep=False):
    
    saveName=-1
    ampHist,c=np.histogram(boutAmps,  bins=8, range=(0,15))
    ampcenters = (c[:-1]+c[1:])/2
    figname='Distribution of Bout Amplitudes. FishID: ' + name
    plt.figure(figname)
    plt.plot(ampcenters, ampHist, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.xlabel('Bout Amplitude (mm/sec)')
    plt.ylabel('Freq.')
    plt.title(figname)
    
    if(save):
        saveName = savepath + name +'_boutAmpsHist.png'
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    return saveName
    
def indAvgBoutFig(avgBout,avgBoutSD,name,savepath,FPS=120,save=True,keep=False):
    
    saveName=-1
    xFr=range(len(avgBout))
    x=np.divide(xFr,FPS)
    plt.figure('avgBout, fish#' + name)
    plt.plot(x,avgBout,linewidth=2)
    plt.ylabel('Velocity mm/s')
    plt.xlabel('Time (s)')
    plt.title('Average Bout +- SE   FishID:' + name)
    negBoutSD=avgBout-avgBoutSD
    posBoutSD=avgBout+avgBoutSD
    
    plt.plot(x,negBoutSD,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,posBoutSD,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,negBoutSD,posBoutSD,alpha=0.2)
    
    if(save):
        saveName=savepath+name+'_avgBout.png'
        plt.savefig(saveName,dpi=600)
    
    if(keep==False):plt.close()
        
    return saveName
    
def indHeatMapFig(heatmap,name,savepath,save=True,keep=False):
    
    saveName=-1
    plt.figure('Heatmap, fish#' + name)
    plt.imshow(heatmap)
    plt.title('Time spent heat map; FishID:' + name)
    
    if(save):
        saveName=savepath+name+'_heatmap.png'
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
        
    return saveName
            
def indCumDistFig(cumDist,name,savepath,FPS=120,save=True,keep=False):
    
    saveName=-1
    xFr=range(len(cumDist))
    x=np.divide(xFr,FPS)
    plt.figure('Cumulative Distance, fish#' + name)
    plt.plot(x,cumDist)
    plt.ylabel('Cumulative Distance mm')
    plt.xlabel('Time (s)')
    plt.title('Cumulative distance travelled; FishID:' + name)
    
    if(save):
        saveName=savepath+name+'_cumDist.png'
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
        
    return saveName
#run(g1,g2,l1,l2,save=False)