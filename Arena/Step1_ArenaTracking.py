# -*- coding: utf-8 -*-
# performs tracking on single fish in behavioural arena
"""
Created on Mon Oct 28 14:34:50 2019

@author: Tom Ryan (Dreosti Lab, UCL)
Adapted from Social Zebrafish workflow by Dreosti-Lab
"""

# Set "Library Path" - Social Zebrafish Repo
lib_path =r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
import sys
sys.path.append(lib_path)

# Import useful libraries
import glob
import numpy as np
import AZ_utilities as AZU
import AZ_video as AZV
import BONSAI_ARK
import timeit
import cv2
import datetime
import pandas as pd
import os

stim=True
#folderListDir='D:\\'
folderListFile ='D:/Movies/RawMovies/aspB2/folderList_210414_aspB2_120Hz.txt'
#folderListFile=folderListDir+folderListFile
plot = True   # set  to true if you want to see the tracking as it happens... this slows the code significantly
FPS = 120
# folder list MUST BE IN THE FOLLOWING FORMAT:

#D:\Arena\191024\
#Blank\
#Dots&Grating\

dateSuff=(datetime.date.today()).strftime("%y%m%d")
AnalysisFolder='D:\\'
AZU.cycleMkDir(AnalysisFolder)
sepTrackingPath=AnalysisFolder + "TrackingData\\" + dateSuff + "\\"
sepTemplatePath=AnalysisFolder + "Templates\\"
if stim:
    sepStimPath=AnalysisFolder + "StimFiles\\" + dateSuff + "\\"
    AZU.cycleMkDir(sepStimPath)
ROI_path, folderNames = AZU.read_folder_list(folderListFile)
AZU.cycleMkDir(sepTrackingPath)
AZU.cycleMkDir(sepTemplatePath)
# grab bonsai files in the ROI_path 
bonsaiFiles = glob.glob(ROI_path + '/*ROI*.bonsai')
if(len(bonsaiFiles)!=0):
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    ROIs = ROIs[:, :]
else:
    ROIs=[]
    
#ROI is in the form ROI[0]=x-origin     N.B. origin (x=0,y=0) in Bonsai is top left
#                   ROI[1]=y-origin
#                   ROI[2]=width
#                   ROI[3]=height
   

for idx,folder in enumerate(folderNames):
# Find all avi files in the folder    
    aviFiles = glob.glob(folder+'/*.avi') 
    if stim:
        stimFiles = glob.glob(folder+'/*.csv') 
        for avi in aviFiles:
            if os.path.exists(avi[0:-4]+'.csv')==False:
                print('Warning! No stim file for ' + avi + ' exists...')
        for stim in stimFiles:
            df = pd.read_csv(stim)
            _,stimName=stim.rsplit(sep='\\',maxsplit=1)
            df.to_csv(sepStimPath+ stimName)
        
                
    # if no aviFiles check for .lnk extensions
    if(len(aviFiles)==0):
        lnkFiles=glob.glob(folder+'/*.lnk')
        if(len(lnkFiles)==0):
            message='No .avi files or .lnk shortcuts in folder #' + folder + '#'
            print(message)
            break

        aviFiles=[]
        for x,lnkFile in enumerate(lnkFiles):
            target,err=AZU.findShortcutTarget(lnkFile)
            if(err==-1):
                message='Could not find target for shortcut #' + lnkFile + '#'
                break

            aviFiles.append(target)
                
    for f,aviFile in enumerate(aviFiles):
        print('Processing ' + aviFile)
        
        # define output path
        d,expName=aviFile.rsplit('\\',1)  # take last part of aviFile path
        expName=expName[0:-4]             # remove the '.avi'
        
        # create a single frame template to use for ROIs later. This way we can archive the original footage after tracking.
#        templateDirPath=d+r'\\Templates\\'
#        AZU.tryMkDir(templateDirPath)
        vid = cv2.VideoCapture(aviFile)
        width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ## grab 5th frame to use as a template for the ROIs
        temp=AZU.grabFrame(aviFile,5)
#        saveName=templateDirPath+expName + '_template.avi'
        saveSepName=sepTemplatePath+expName + '_template.avi'
#        out = cv2.VideoWriter(saveName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
        outSep = cv2.VideoWriter(saveSepName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
#        out.write(temp)
        outSep.write(temp)
#        out.release()
        outSep.release()
        vid.release()
        
        if(f==0):
            figureDirPath=d+r'\Figures'
            trackingDirPath=d+r'\Tracking'
            
            AZU.tryMkDir(figureDirPath)
            AZU.tryMkDir(trackingDirPath)
            
        # run the tracking 
        timeNow=timeit.time.ctime()
        print('Tracking started on ' + timeNow)
        tic=timeit.default_timer()
        
        try:
            fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, err,errF = AZV.arena_fish_tracking(aviFile, figureDirPath, ROIs, plot=1, cropOp=1, FPS=FPS)
        except:
            print('aviFile failed during tracking, skipping...')
            err=True

        toc=timeit.default_timer()
        timeNow=timeit.time.ctime()
        message='Tracking finished on ' + timeNow
        # Judge success (less than 10% of the movie were not 'No Contour' or 'Particle too small' frames) 
#        not written yet
        # Create shortcuts in seperate folders for easy access to .avis for future analysis
        if(err):
#            message=message + '. But failed on frame ' + str(errF)
            saveFailPath=d+r'\\failedAviFiles\\' + expName + '.lnk'
#            AZU.createShortcut(aviFile,saveFailPath)
        else:
            message=message + '. Appears successful!'
            saveSuccessPath=d+r"\\FinishedAviFiles\\" + expName + '.lnk'
#            AZU.createShortcut(aviFile,saveSuccessPath)
            
            print(message)
            message='Took ' + str(toc-tic) + ' seconds to process'
            print(message)
        
            fish = np.vstack((fxS[:,0], fyS[:,0], bxS[:,0], byS[:,0], exS[:,0], eyS[:,0], areaS[:,0], ortS[:,0], motS[:,0]))
            # Save tracking for each file in it's own folder
            filename=trackingDirPath + '\\' + expName + '_tracking.npz'
            print('Saving tracking at ' + filename)
            np.savez(filename, tracking=fish.T)
            # and in seperate tracking folder
            trackname=sepTrackingPath+ expName + '_tracking.npz'
            print('Saving tracking at ' + trackname)
            np.savez(trackname, tracking=fish.T)
    #        AZU.createShortcutTele(filename)
        