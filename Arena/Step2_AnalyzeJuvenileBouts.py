# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:25:27 2021

@author: thoma
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Arena Zebrafish Repo
lib_path = r'C:\Users\Tom\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------
# Set Library Paths
import sys
sys.path.append(lib_path)
lib_path = r'C:\Users\Tom\Documents\GitHub\Arena_Zebrafish\ARK\libs'
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import datetime
import glob
# Import local modules
import AZ_utilities as AZU
import ARK_bouts_new as ARKB

folderListFile=[]
#trackingFolder=[]
folder=r'E:\dataGautam\Juv\B0\210924\Sham'
trackingFolder=folder + r'\\Tracking\\'
#DictionaryFolder=folder + r'\\Analysis\\Dictionaries\\'
#figureFolder=folder + r'\\Analysis\\Figures\\'
#AZU.cycleMkDir(DictionaryFolder)
#AZU.cycleMkDir(figureFolder)
# Set FPS of camera
FPS = 120
# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile =r'F:\Data\Movies\Loom+Para\Loom+Para.txt'

# OR

#trackingFolder = r'D:\\Movies\\GroupedData\\Groups\\para\\'
#suff='WT_R0_201115' ## Suffix to add to end of Analysis dictionary name and analysis folder
dateSuff=(datetime.date.today()).strftime("%y%m%d")
#suff=''
sf=0*60*FPS
ef=-1

trackingFiles=[]
dictList=[]
dictNameList=[]
#    groupName=nn + pstr

# Grab tracking files from folder or .txt folder list file
#trackingFiles=AZU.getTrackingFilesFromFolder(suff=suff,folderListFile=folderListFile,trackingFolder=trackingFolder)
# Or just search the TrackingData Directory
trackingFiles=glob.glob(trackingFolder+'*tracking.npz')
tailXFiles=glob.glob(trackingFolder+'*SegX.csv')
tailYFiles=glob.glob(trackingFolder+'*SegY.csv')

# run through each experiment
for k,trackingFile in enumerate(trackingFiles):
    # load data
    wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
    wDir=wDir+r'\\Analysis\\'
    AZU.cycleMkDir(wDir)
    fx,fy,bx,by,ex,ey,area,ort,_=AZU.grabTrackingFromFile(trackingFile)
    
    # analyse bouts and save
    bouts=ARKB.analyze(trackingFile,path=True)
    savepath=wDir+name+'_bouts'
    print('Saving bouts at ' + savepath)
    np.save(savepath,bouts)
    
    # analyse tail segment angles and save
    cumulAngleS=[]
    finalSegAngleS=[]
    deltaThetasS=[]
    curvatureS=[]
    # read tailX and tailY files
    xData=np.genfromtxt(tailXFiles[k], delimiter=',')
    yData=np.genfromtxt(tailYFiles[k], delimiter=',')
    
    # rotate tail trajectory in the same way you would a train of x and y coordinates (since that's what it is)
#    print('Rotating tail trajectories')
#    tailX,tailY=AZA.rotateTrajectoriesByHeadings(xData,yData,ort)
    
    # Determine number of frames, segments and angles
    num_frames = np.shape(xData)[0]-1
    num_segments = np.shape(xData)[1]
    num_angles = num_segments - 1
      
    # Allocate space for measurements - creates empty arrays to store data
    cumulAngles = np.zeros(num_frames)
    curvatures = np.zeros(num_frames)
    finalSegAngle = np.zeros(num_frames)
    bodyTheta = np.zeros([num_frames,num_angles])
    ## Measure tail motion, angles, and curvature ##
    # Take x and y values of first frame for each segment
    prev_xs = xData[0, :]
    prev_ys = yData[0, :]
    
    ########### Start of frame loop (f) #################
    for f in range(num_frames):
#        print(str(f))
        delta_thetas = np.zeros(num_angles) # Make an array of zeros with the same size as num angles (num seg-1)
        
        # find angle between eyes and body (heading again)
        dx = bx[f] - ex[f]  
        dy = by[f] - ey[f] 
        heading = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi)
        
        prev_theta = heading # set first theta to heading
        # measure angle between body and last segment
        dx = xData[f, -1] - bx[f] 
        dy = yData[f, -1] - by[f]
        theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
        if np.abs(theta - prev_theta)>180:
            theta*=-1
            
        finalSegAngle[f] = theta-prev_theta
        ############### Start of segment loop (a) ################
        for a in range(num_angles):
            ddx = xData[f, a] - bx[f]
            ddy = yData[f, a] - by[f]
            thetaa = np.arctan2(ddy, ddx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
            if np.abs(thetaa - heading)>180:
                thetaa*=-1
            bodyTheta[f, a] = thetaa - heading
            
            dx = xData[f, a+1] - xData[f, a] # dx between each segment for the same frame
            dy = yData[f, a+1] - yData[f, a] # dy between each segment for the same frame
            theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
            if np.abs(theta - prev_theta)>180:
                theta*=-1
            delta_thetas[a] = theta - prev_theta
            prev_theta = theta # prev theta is set to current theta
        ############### End of angle loop (a) ################
        
        cumulAngles[f] = np.sum(delta_thetas) # sum all angles for this frame
        curvatures[f] = np.mean(np.abs(delta_thetas)) # mean of abs value of angles
        
        # Store previous tail
        prev_xs = xData[f, :] # So that we don't always take the 1st frame of the movie to calculate motion
        prev_ys = yData[f, :]
    ####### End of frame loop (f) ###############
    tail=np.zeros([num_frames,3])
    tail[:,0]=finalSegAngle
    tail[:,1]=cumulAngles
    tail[:,2]=curvatures
    # stack and save
    filename=wDir+name+'_tailAnalysis'
    print('Saving tailAnalysis at ' + filename)
    np.save(filename, tail)
    filename=wDir+name+'_bodyTheta'
    np.save(filename,bodyTheta)