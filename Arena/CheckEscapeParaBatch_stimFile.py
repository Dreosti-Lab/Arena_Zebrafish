# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:15:32 2021

@author: thoma
"""
import AZ_analysis as AZA
import AZ_utilities as AZU
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# plot figures for individual and/or all fish?
plotInd=False
plotSum=True

#folderListFile_1=[]
#folderListFile_2=[]
## OR ##
trackingFolders=[] 
trackingFolders.append(r'D:\\Movies\\RawMovies\\ctrlB2\\allTrackingToGroup')
trackingFolders.append(r'D:\\Movies\\RawMovies\\aspB2\\allTrackingToGroup')
stimFolder=r'D:\\StimFiles\\'
templateFolder=r'D:\\Templates\\'
# Define colors to use for different conditions
colorMaps=['Greens','Oranges']

# create lists to store collected fish data for each condition
trajXSS=[]
trajYSS=[]
rotTrajXSS=[]
rotTrajYSS=[]
respProbSS=[]
resSS=[]
booSS=[]
AvCumDistSS=[]
AvMaxDistPerFrameActSS=[]
AvMaxMotionSS=[]

## Thresholds set empirically observing differences between normal bouts and escape responses
motionThresh=0.23
cumDistThresh=5
distPerFrameThresh=1.2

for k in range(0,len(trackingFolders)):           
    trackingFiles=glob.glob(trackingFolders[k]+'\*.npz')
    if k==0:
        trackingFiles=trackingFiles[2:]
    # create lists to store collected fish data
    trajXS=[]
    trajYS=[]
    rotTrajXS=[]
    rotTrajYS=[]
    respProbS=[]
    resS=[]
    booS=[]
    AvCumDistS=[]
    AvMaxDistPerFrameActS=[]
    AvMaxMotionS=[]

    ## Input list of paramters if mixed data in the loop: length of lists must == len(trackingFiles). If only one parameter given it will be applied for all experiments
    stS=[15]
    inS=[1]
    duS=[1]
    frameRate=[100]
    
    ## (usually) constant parameters (can be extended also to individual experiments)
    responseSeconds=0.5 # number of seconds to search for a response
    numSecs=[1] # number of seconds to measure for escape response properties
    windowSeconds=3 # the number of seconds to plot escape responses (e.g. average bout)

    if len(stS)!=len(inS) or len(stS)!=len(duS) or len(inS)!=len(duS) or len(stS)!=len(trackingFiles): print ('Your parameter lists are not the same length as each other or the number of trackingFiles')
    # loop through files
    for i,trackingFile in enumerate(trackingFiles): 
        
        # find display image
        wDir,name,_,_,_,_,_=AZU.grabFishInfoFromFile(trackingFile)
        stimFile=stimFolder+name+'.csv'
        img=AZU.grabFrame(templateFolder+name+'_template.avi',0)
        # grab parameters for this iteration
        if len(frameRate)==1:frR=frameRate[0]
        else:frR=frameRate[i]
        if len(numSecs)==1: nS=numSecs[0]
        else: nS=numSecs[i]
        nFr=nS*frR
    
        if len(stS)==1:st=stS[0]
        else:st=stS[i]
        if len(inS)==1:inte=inS[0]
        else: inte=inS[i]
        if len(duS)==1: du=duS[0]
        else: du=duS[i]
        
        fx,fy,_,_,_,_,_,ort,motion=AZU.grabTrackingFromFile(trackingFile)
        fx_mm,fy_mm=AZU.convertToMm(fx,fy)
        distPerFrame,cumDist=AZU.computeDistPerFrame(fx_mm,fy_mm)
    
#        loomStarts,loomEnds,respEnds,_=AZA.findLooms(len(fx),startTime=st,interval=inte,duration=du,numFrames=nFr,frameRate=frR,responseSeconds=responseSeconds)
        # Start by using the loom stim file to find when the stims started, and where exactly they were
        loomStarts,loomEnds,respEnds,loomPosX,loomPosY=AZU.findLoomsFromFile(stimFile,responseSeconds=responseSeconds,windowSeconds=windowSeconds,FPS=frameRate) 
        # Now grab the trajectories. trajXOrig is the list of raw trajectories, trajX has been subtracted so they all have a common origin, trajHeadings are the original heading of the animal when the loom initiated
        trajX,trajY,trajHeadings,trajXOrig,trajYOrig,=AZA.extractTrajFromStim(loomStarts,loomEnds,fx,fy,ort)
        # Now rotate the trajectories so they all have a common starting heading
        rotTrajX,rotTrajY=AZA.rotateTrajectoriesByHeadings(trajXOrig,trajYOrig,trajHeadings)
    
        # loop through looms to determine whether we consider the behaviour in this window a 'response' or not
        # Note that this utilised the 'respEnds' which is a shorter time interval than the loomEnds, set at 3 seconds for display of bout and trajectory.
        # RespEnds currently set at 500ms (in AZU.findLoomsFromFile)
        res=[]
        motionAct=[]
        cumDistAct=[]
        distPerFrameAct=[]
        
        for j in range(0,len(loomStarts)):
            # Test whether we think this is an escape by finding maximum motion signal
            motionAct.append(np.max(motion[loomStarts[j]:respEnds[j]]))
            # cumulative distance
            cumDistAct.append(cumDist[respEnds[j]]-cumDist[loomStarts[j]])
            # and max velocity 
            distPerFrameAct.append(np.max(distPerFrame[loomStarts[j]:respEnds[j]]))
            
            # ALL must be over threshold to be considered an escape response, and all are set at thresholds higher than an average bout, though not explicitly.
            print('For Loom #' + str(j))
                  #    if motionAct>motionThresh:
                  #        a=1
                  ##        print('Motion - TRUE')
                  #    else:
                  #        a=1
                  #        print('Motion - FALSE')
                
            if cumDistAct[j]>cumDistThresh:
                a=1
                #        print('cumDist - TRUE')
            else:
                a=1
                #        print('cumDist - FALSE')
                
            if distPerFrameAct[j]>distPerFrameThresh:
                a=1
                #        print('distPerFrame - TRUE')
            else:
                a=1
                #        print('distPerFrame - FALSE')
            if (cumDistAct[j]<cumDistThresh and distPerFrameAct[j]<distPerFrameThresh): 
                print('UNANIMOUS! FALSE')
                res.append(0)
            elif (cumDistAct[j]>cumDistThresh and distPerFrameAct[j]>distPerFrameThresh):
                print('UNANIMOUS! TRUE')
                res.append(1)
            else:
                print('Inconsistent results...')
                res.append(0)
            # END OF LOOM LOOP
            
        res_sm=AZM.smoothSignal(res,3)
        boo=np.asarray(res)>0
     
        if plotInd:
            plt.figure()
            plt.plot(res_sm)
            plt.xlim(2,27)
    
            if np.sum(boo)>0:
                plt.figure()
                for i in range(0,len(res)):
                    if(boo[i]):plt.plot(rotTrajY[i],rotTrajX[i])
                
            plt.figure()
            plt.imshow(img*-1,cmap='Greys')
            for i in range(0,len(res)):
                if boo[i]:plt.plot(trajX[i],trajY[i])
    
        # collect booleans of escape detection for this fish
        booS.append(boo)
                
        ## Collect escape trajectories for each fish, untouched and unrotated....
        trajXS.append(trajX)
        trajYS.append(trajY)
        
        ## ... and transformed and rotated...
        rotTrajXS.append(rotTrajX)
        rotTrajYS.append(rotTrajY)
        
        ## Compute and collect overall probability of response to the loom over all looms (will need to exclude those that were not really seen by counting the looms that occured when the fish was very near the edge of the chamber)
        respProbS.append(np.mean(boo))
    
        ## compute and collect the array containing each loom's response (true or false)
        resS.append(res)
        
        # For each fish, compute the averages of their loom's...
        # Total Distance
        AvCumDistS.append(np.mean(cumDistAct))
        # Max Velocity
        AvMaxDistPerFrameActS.append(np.max(distPerFrameAct))
        # Max Motion
        AvMaxMotionS.append(np.max(motionAct))
        # END OF FISH LOOP
    
    ## Collect array of fish for this group
    # collect booleans of escape detection
    booSS.append(booS)
    ## Collect escape trajectories for each fish, untouched and rotated
    trajXSS.append(trajXS)
    trajYSS.append(trajYS)
    rotTrajXSS.append(rotTrajXS)
    rotTrajYSS.append(rotTrajYS)
    ## Compute and collect overall probability of response to the loom over all looms (will need to exclude those that were not really seen by counting the looms that occured when the fish was very near the edge of the chamber)
    respProbSS.append(respProbS)
    ## compute and collect the array containing each loom's response (true or false)
    resSS.append(resS)
    
    #  Collect for this group for comparisons: 
    # Total Distance
    AvCumDistSS.append(AvCumDistS)
    # Max Velocity
    AvMaxDistPerFrameActSS.append(AvMaxDistPerFrameActS)
    # Max Motion
    AvMaxMotionSS.append(AvMaxMotionS)
    
    if plotSum:
        #    plt.figure()
        numFish=len(rotTrajXS)
        cols = cm.get_cmap(colorMaps[k], 128)
        colIter=np.linspace(0.25,0.75,num=numFish)
        countTotalLooms=0 
        for i in range(0,numFish):
            numLooms=len(rotTrajXS[i])
            for j in range(0,numLooms):
                if booS[i][j]:
                    countTotalLooms+=1
                    plt.plot(rotTrajYS[i][j],rotTrajXS[i][j],c=cols(colIter[i]),alpha=0.5)
    print('For the ' + str(k) + 'st group, I detected a total of ' + str(countTotalLooms)+ ' escapes from ' + str(numFish) + ' fish')
    # END OF GROUP LOOP
    
#if plotAll:
#    
#    plt.figure('Escape trjectory Parameters')

# create a swarm plot to show different parameters
    
 
#milan= (73, 43, 44, 70, 61)
#inter = (54, 59, 69, 46, 58)
#fig, ax = plt.subplots()
#index = np.arange(n)
#bar_width = 0.35
#opacity = 0.9
#ax.bar(index, milan, bar_width, alpha=opacity, color='r',
#                label='Milan')
#ax.bar(index+bar_width, inter, bar_width, alpha=opacity, color='b',
#                label='Inter')
#ax.set_xlabel('Seasons')
#ax.set_ylabel('Points')
#ax.set_title('Milan v/s Inter')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('1995-96','1996-97','1997-98','1998-99','1999-00'
#    ))
#ax.legend()
#plt.show()
    
    