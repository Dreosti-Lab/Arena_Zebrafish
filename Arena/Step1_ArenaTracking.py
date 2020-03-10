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


folderListFile =r'S:\WIBR_Dreosti_Lab\Tom\Data\Movies\FolderLists\200308_AllExpReverseOrder.txt'
plot = 1    # set  to 1 if you want to see the tracking as it happens... this slows the code significantly
# folder list MUST BE IN THE FOLLOWING FORMAT:
# include a space at the end of the first line

#D:\Arena\191024\ 
#Arena\Blank
#Arena\Dots&Grating

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
   

for idx,folder in enumerate(folderNames):
# Find all avi files in the folder    
    aviFiles = glob.glob(folder+'/*.avi') 
    for f,aviFile in enumerate(aviFiles):
        
        # define output path
        d,expName=aviFile.rsplit('\\',1)  # take last part of aviFile path
        expName=expName[0:-4]             # remove the '.avi'
        if(f==0):
            figureDirPath=d+r'\FiguresCrop'
            trackingDirPath=d+r'\TrackingCrop'
            AZU.tryMkDir(figureDirPath)
            AZU.tryMkDir(trackingDirPath)
                    
        # run the tracking 
        tic=timeit.default_timer()
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = AZV.arena_fish_tracking(aviFile, figureDirPath, ROIs)
        toc=timeit.default_timer()
        message='Took ' + str(toc-tic) + ' seconds to process'
        print(message)
        # Save tracking for each file in it's own folder
        filename=trackingDirPath + r'\\' + expName + '_tracking.npz'
        fish = np.vstack((fxS[:,0], fyS[:,0], bxS[:,0], byS[:,0], exS[:,0], eyS[:,0], areaS[:,0], ortS[:,0], motS[:,0]))
        np.savez(filename, tracking=fish.T)