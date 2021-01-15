# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:11:26 2021

@author: thoma
"""
import AZ_analysis as AZA

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

loomStarts,loomEnds=AZA.findLooms(movieLengthFr,
                              startTime=15,
                              interval=2,
                              duration=1,
                              numFrames=60,
                              frameRate=120)

trajectoriesX,trajectoriesY,trajectoriesHeadings=AZA.extractTrajFromStim(loomStarts,loomEnds,fx,fy,heading)
rotTrajX,rotTrajY=AZA.rotateTrajectoriesByHeadings(trajectoriesX,trajectoriesY,trajectoriesHeadings)

[fx_mm],[fy_mm]=AZU.convertToMm([fx],[fy])

distPerFrame,cumDist=AZU.computeDistPerFrame(fx_mm,fy_mm)