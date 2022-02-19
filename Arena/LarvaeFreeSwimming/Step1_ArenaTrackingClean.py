# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:18:37 2022

@author: Tom
"""
## Clean version of Step1_ArenaTracking script Feb 2022
## Given a folderListFile consisting of folder paths to raw movies, this script 
## will search the folders for movies, then track individual fish. Outputs are 
## saved in Figures, Tracking and Movie folders defined by the user and partly 
## inferred from the name of the movie (chamber and condition codes) 

## Settings
stim=False
tailTracking=True # set to true to segment and track tail
saveCroppedMovie=True # set to true if you want to save a cropped version of the movie following tracking
plot = True   # set  to true if you want to see the tracking as it happens... this slows the code significantly
FPS = 120
cropSize=[256,256]
larvae=True
Chamber='B0'
Condition='ControlTest' 
inPlace=False        # Set to true to save outputs in associated folders with the raw data
sepPlace=True       # Set to true to save outputs in seperate analysis folder (given below)
AnalysisFolder='D:\\Analysis\\'

## Input folderListFile
folderListFile='S:/WIBR_Dreosti_Lab/Tom/Data/LarvaeFreeSwimming_NEW/EA_B0_LarvaeFreeSwimming.txt'

###############################################################################
# Set "Library Path" - Arena Zebrafish Repo
lib_path =r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
import sys
sys.path.append(lib_path)

# Import useful libraries
import glob
import numpy as np
import AZ_utilities as AZU
import AZ_video as AZV
import BONSAI_ARK
import timeit
import datetime
import pandas as pd
import os

## Define and create input and oputput paths

dateSuff=(datetime.date.today()).strftime("%y%m%d")
AZU.cycleMkDirr(AnalysisFolder)

if larvae:
    fold=AnalysisFolder+'LarvaeFreeSwimming_NEW\\'
else:
    fold=AnalysisFolder+'JuvenileFreeSwimming\\'

if sepPlace:
    sepMoviePath=fold + 'CroppedMovies\\' + Chamber + '\\' + Condition + '\\'    
    sepTrackingPath=fold + 'TrackingData\\' + Chamber + '\\' + Condition + '\\'
    sepFigurePath=fold + 'TrackingFigures\\' + Chamber + '\\' + Condition + '\\'
    sepTemplatePath=fold + 'Templates\\' + Chamber + '\\' + Condition + '\\'
    
    AZU.cycleMkDirr(sepFigurePath)
    AZU.cycleMkDirr(sepMoviePath)
    AZU.cycleMkDirr(sepTrackingPath)
    AZU.cycleMkDirr(sepTemplatePath)
    
    if stim:
        sepStimPath=fold + 'StimFiles\\' + Chamber + '\\' + Condition + '\\'
        AZU.cycleMkDirr(sepStimPath)

ROI_path, folderNames = AZU.read_folder_list(folderListFile)
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

## Search through all folders and create a list of all aviFiles and stimFiles if needed
aviFilesS=[]
if stim:
    stimFilesS=[]

for idx,folder in enumerate(folderNames):
# Find all avi files in the folder    
    aviFiles = glob.glob(folder+'/*.avi') 
    # if a stimulus is given, search the same folder for stim csv files
    if stim:
        stimFiles = glob.glob(folder+'/*.csv') 
        for avi in aviFiles:
            if os.path.exists(avi[0:-4]+'.csv')==False:
                print('Warning! No stim file for ' + avi + ' exists...')
        for stim in stimFiles:
            df = pd.read_csv(stim)
            _,stimName=stim.rsplit(sep='\\',maxsplit=1)
            df.to_csv(sepStimPath+ stimName)
            stimFilesS.append(stim)
            
    # if no aviFiles check for .lnk extensions and add their targets to the list
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
    # add this folders aviFiles to the list
    aviFilesS.append(aviFiles)
    
# flatten list (so we don't have a list of lists)
aviFiles=[]
aviFiles = [file for files in aviFilesS for file in files]

# Now loop through each aviFile
failedAviFiles=[]
for f,aviFile in enumerate(aviFiles):
    print('Processing ' + aviFile)
    
    # define output path
    d,expName=aviFile.rsplit('\\',1)  # take last part of aviFile path
    expName=expName[0:-4]             # remove the '.avi'
    
    # create a single frame template to use for ROIs later. 
    if inPlace:
        templateDirPath=d+r'\\Templates\\'
        AZU.tryMkDir(templateDirPath)
        templatePath=AZV.createSingleFrameVid(aviFile,templateDirPath,expName)
    if sepPlace:
        templatePath=AZV.createSingleFrameVid(aviFile,sepTemplatePath,expName)
        
    # run the tracking 
    timeNow=timeit.time.ctime()
    print('Tracking started on ' + timeNow)
    tic=timeit.default_timer()
    
    try:
        fxS, fyS, bxS, byS, exS, eyS, tailSegXS, tailSegYS, areaS, ortS, motS = AZV.arena_fish_tracking(aviFile, ROIs, sepFigurePath=sepFigurePath, sepMoviePath=sepMoviePath, maxNumFrames=(10)*FPS, FPS=FPS, plot=True, cropOp=True, saveCroppedMovie=True, display=False, trackTail=True, larvae=larvae, inPlace=inPlace, sepPlace=sepPlace)
    
        toc=timeit.default_timer()
        timeNow=timeit.time.ctime()
        message='Tracking finished on ' + timeNow
        print(message)
        print(message)
        message='Took ' + str(toc-tic) + ' seconds to process'
        print(message)
    
        fish = np.vstack((fxS[:,0], fyS[:,0], bxS[:,0], byS[:,0], exS[:,0], eyS[:,0], areaS[:,0], ortS[:,0], motS[:,0]))
        
        if inPlace:
            trackingDirPath=d+'\\Tracking'
            fileRoot = trackingDirPath + '\\' + expName
            filename = fileRoot + '_tracking.npz'
            np.savez(filename, tracking=fish.T)
        if tailTracking:
            pd.DataFrame(tailSegXS).to_csv(fileRoot + '_TailSegX.csv', header=None, index=None)
            pd.DataFrame(tailSegYS).to_csv(fileRoot + '_TailSegY.csv', header=None, index=None)
        
        if sepPlace:
            fileRoot = sepTrackingPath + '\\' + expName
            filename = fileRoot + '_tracking.npz'
            np.savez(filename, tracking=fish.T)
        if tailTracking:
            pd.DataFrame(tailSegXS).to_csv(fileRoot + '_TailSegX.csv', header=None, index=None)
            pd.DataFrame(tailSegYS).to_csv(fileRoot + '_TailSegY.csv', header=None, index=None)  
            
    except:
        print('aviFile failed during tracking, skipping...')
        failedAviFiles.append(aviFile)
    
