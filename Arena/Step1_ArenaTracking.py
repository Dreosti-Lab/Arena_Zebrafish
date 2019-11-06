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

folderListFile = r'D:\Arena\FolderLists\191024.txt'

# folder list MUST BE IN THE FOLLOWING FORMAT:
# include a space at the end of the first line

#D:\Arena\191024\ 
#Arena\Blank
#Arena\Dots&Grating

ROI_path, folderNames = AZU.read_folder_list(folderListFile)

# grab bonsai files in the ROI_path 
bonsaiFiles = glob.glob(ROI_path + '/*.bonsai')
bonsaiFiles = bonsaiFiles[0]
ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
ROIs = ROIs[:, :]
#ROI is in the form ROI[0]=x-origin
#                   ROI[1]=y-origin
#                   ROI[2]=width
#                   ROI[3]=height
   

for idx,folder in enumerate(folderNames):
# Find all avi files in the folder    
    aviFiles = glob.glob(folder+'/*.avi') 
    for f,aviFile in enumerate(aviFiles):
        
        # define output path
        d,expName=aviFile.rsplit('\\',1)  # take last part of aviFile path
        expName=expName[0:-4]             # remove the '.avi'
        if(f==0):
            figureDirPath=d+'\\Figures'
            trackingDirPath=d+'\\Tracking'
            AZU.tryMkDir(figureDirPath)
            AZU.tryMkDir(trackingDirPath)
                    
        # run the tracking 
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = AZV.arena_fish_tracking(aviFile, figureDirPath, ROIs)
        
        # Save tracking for each file in it's own folder
        filename=trackingDirPath + '\\' + expName + '_tracking.npz'
        fish = np.vstack((fxS[:], fyS[:], bxS[:], byS[:], exS[:], eyS[:], areaS[:], ortS[:], motS[:]))
        np.savez(filename, tracking=fish.T)