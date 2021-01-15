# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:11:26 2021

@author: thoma
"""
import AZ_analysis as AZA
import AZ_utilities as AZU
#fx
#fy
#heading
#movieLengthFr

# startTime in minutes, length of adaptation
# interval in minutes, time between looms
# duration in seconds, duration of stim (since stimulus ENDS on the marker time, so starts marker-duration)
# movieLengthFr is length of full movie in frames including adaptation 
# numFrames is the number of frames to extract for trajectory analysis
# frameRate is camera frame rate in Hz
trackingSHDir = r'D:\\Movies\\GroupedData\\Groups\\'
suffs=[r'EC_B1',r'EA_B1',r'WT_B1']


trackingSHFolders =[]
for i in range(0,len(suffs)):
    trackingSHFolders.append(trackingSHDir + suffs[i])


rotTrajXS=[]
rotTrajYS=[]
escapes_motionS=[]
escapes_ortS=[]
escapes_distPerFrameS=[]

for bigLoop,trackingSHFolder in enumerate(trackingSHFolders):
    trackingFiles=AZU.getTrackingFilesFromFolder(trackingFolder=trackingSHFolder)
    
    for trackingFile in trackingFiles:
        
        fx,fy,_,_,_,_,_,ort,motion=AZU.grabTrackingFromFile(trackingFile)
        loomStarts,loomEnds=AZA.findLooms(len(fx),
                                          startTime=15,
                                          interval=2,
                                          duration=1,
                                          numFrames=60,
                                          frameRate=120)

        [fx_mm],[fy_mm]=AZU.convertToMm([fx],[fy])
        trajX,trajY,trajHeadings=AZA.extractTrajFromStim(loomStarts,loomEnds,fx_mm,fy_mm,ort)
        rotTrajX,rotTrajY=AZA.rotateTrajectoriesByHeadings(trajX,trajY,trajHeadings)
        distPerFrame,cumDist=AZU.computeDistPerFrame(fx_mm,fy_mm)
        [escapes_motion,escapes_ort,escapes_distPerFrame]=AZA.extractVecFromStim([loomStarts,loomStarts,loomStarts],[loomEnds,loomEnds,loomEnds],[motion,ort,distPerFrame])
        rotTrajXS.append(rotTrajX)
        rotTrajYS.append(rotTrajY)
        escapes_motionS.append(escapes_motion)
        escapes_ortS.append(escapes_ort)
        escapes_distPerFrameS.append(escapes_distPerFrame)
        