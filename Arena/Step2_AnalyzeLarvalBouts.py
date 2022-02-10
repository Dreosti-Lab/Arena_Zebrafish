# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:25:27 2021

@author: thoma
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Arena Zebrafish Repo
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------
# Set Library Paths
import sys
sys.path.append(lib_path)
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\GitHub\Arena_Zebrafish\ARK\libs'
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import datetime
import glob
# Import local modules
import AZ_utilities as AZU
import ARK_bouts as ARKB

FPS = 120
# folderListFile='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Sham.txt'
# folderListFile='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Lesion.txt'
# dateSuff=(datetime.date.today()).strftime("%y%m%d")
# sf=0*60*FPS
# ef=-1

# _,folderNames = AZU.read_folder_list(folderListFile)
# trackingFiles,tailXFiles,tailYFiles=[],[],[]
# for folder in folderNames:
#     trackingFolder=folder + r'\\Tracking\\'
#     # Grab tracking files from folder or .txt folder list file
#     trackingFilest=glob.glob(trackingFolder+'*tracking.npz')
#     tailXFilest=glob.glob(trackingFolder+'*SegX.csv')
#     tailYFilest=glob.glob(trackingFolder+'*SegY.csv')
#     for i,s in enumerate(trackingFilest): #
#         trackingFiles.append(s)
#         tailXFiles.append(tailXFilest[i])
#         tailYFiles.append(tailYFilest[i])
trackingFolders=['S:\WIBR_Dreosti_Lab\Tom\DataForAdam\GroupedTracking\EC_B0','S:\WIBR_Dreosti_Lab\Tom\DataForAdam\GroupedTracking\EA_B0','S:\WIBR_Dreosti_Lab\Tom\DataForAdam\GroupedTracking\EC_M0','S:\WIBR_Dreosti_Lab\Tom\DataForAdam\GroupedTracking\EA_M0']
for trackingFolder in trackingFolders:
    trackingFiles=glob.glob(trackingFolder+'\*tracking*.npz')
    # run through each experiment
    for k,trackingFile in enumerate(trackingFiles):
        # load data
        wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
        wDir=wDir+r'\\Analysis\\'
        AZU.cycleMkDir(wDir)
        fx,fy,bx,by,ex,ey,area,ort,_=AZU.grabTrackingFromFile(trackingFile)
        # tailXFile=tailXFiles[k]
        # tailYFile=tailYFiles[k]
        
        # analyse bouts and save
        bouts=ARKB.analyze(trackingFile,path=True)
        savepath=wDir+name+'_bouts'
        print('Saving bouts at ' + savepath)
        np.save(savepath,bouts)
    
    # # analyse tail segment angles and save
    # cumulAngleS=[]
    # deltaThetasS=[]
    # curvatureS=[]
    # read tailX and tailY files
    # xData=np.genfromtxt(tailXFiles[k], delimiter=',')
    # yData=np.genfromtxt(tailYFiles[k], delimiter=',')
    
    # Determine number of frames, segments and angles
#     num_frames = np.shape(xData)[0]-1
#     num_segments = np.shape(xData)[1]
#     num_angles = num_segments - 1
      
#     # Allocate space for measurements - creates empty arrays to store data
#     cumulAngles = np.zeros(num_frames)
#     curvatures = np.zeros(num_frames)
#     bodyTheta = np.zeros([num_frames,num_angles])
#     ## Measure tail motion, angles, and curvature ##
#     # Take x and y values of first frame for each segment
#     prev_xs = xData[0, :]
#     prev_ys = yData[0, :]
    
#     ########### Start of frame loop (f) #################
#     for f in range(num_frames):
# #        print(str(f))
#         delta_thetas = np.zeros(num_angles) # Make an array of zeros with the same size as num angles (num seg-1)
        
#         # find angle between eyes and body (heading again)
#         dx = bx[f] - ex[f]  
#         dy = by[f] - ey[f] 
#         heading = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi)
        
#         prev_theta = heading # set first theta to heading
#         # measure angle between body and last segment
#         dx = xData[f, -1] - bx[f] 
#         dy = yData[f, -1] - by[f]
#         theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
#         if np.abs(theta - prev_theta)>180:
#             theta*=-1
#         ############### Start of angle loop (a) ################
#         for a in range(num_angles):
#             ddx = xData[f, a] - bx[f]
#             ddy = yData[f, a] - by[f]
#             thetaa = np.arctan2(ddy, ddx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
#             if np.abs(thetaa - heading)>180:
#                 thetaa*=-1
#             bodyTheta[f, a] = thetaa - heading
            
#             dx = xData[f, a+1] - xData[f, a] # dx between each segment for the same frame
#             dy = yData[f, a+1] - yData[f, a] # dy between each segment for the same frame
#             theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
#             if np.abs(theta - prev_theta)>180:
#                 theta*=-1
#             delta_thetas[a] = theta - prev_theta
#             prev_theta = theta # prev theta is set to current theta
#         ############### End of angle loop (a) ################
        
#         cumulAngles[f] = np.sum(delta_thetas) # sum all angles for this frame
#         curvatures[f] = np.mean(np.abs(delta_thetas)) # mean of abs value of angles
        
#         # Store previous tail
#         prev_xs = xData[f, :] # So that we don't always take the 1st frame of the movie to calculate motion
#         prev_ys = yData[f, :]
        
#     ####### End of frame loop (f) ###############
#     tail=np.zeros([num_frames,2])
#     tail[:,0]=cumulAngles
#     tail[:,1]=curvatures
#     # stack and save
#     filename=wDir+name+'_tailAnalysis'
#     print('Saving tailAnalysis at ' + filename)
#     np.save(filename, tail)
#     filename=wDir+name+'_bodyTheta'
#     np.save(filename,bodyTheta)