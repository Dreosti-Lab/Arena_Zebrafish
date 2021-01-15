# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:46:22 2020

@author: thoma
"""

# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'

import sys
sys.path.append(lib_path)
import numpy as np
import AZ_utilities as AZU
import AZ_summary as AZS
import AZ_analysis_testing as AZA
import glob
import AZ_ROITools as AZR
import AZ_streakProb as AZP
import cv2

FPS=120

folderListFile=[]
trackingFolder=[]
trackingSHFolder=[]

# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile = r'D:\Movies\FolderLists\AllCtrl.txt'

# OR

trackingSHFolder = r'D:\\Movies\\GroupedData\\Groups\\EmxGFP_Asp_Damage\\'

# Set Flags
createSpatialFigures=True
keepSpatialFigures=True
sameROIs = True

#ROINames=['RestrictedCentre']
ROINames=['Top Left','Top','Top Right','Middle Left','Central Chamber', 'Middle Right', 'Bottom Left','Bottom','Bottom Right']
numROIs=len(ROINames)

trackingFiles=[]
trackingNameList=[]
trackingList=[]
aviNameList=[]
opt=-1
## Compile list of tracking files based on input folders. Also build list of corresponding movies to extract a few frames and draw ROIs
if(len(folderListFile)!=0 and len(trackingSHFolder)==0): # then we are dealing with a folderList rather than a folder of shortcuts
    opt=1
    # Read Folder List
    _,folderNames = AZU.read_folder_list(folderListFile)

    # Bulk analysis of all folders
    for idx,folder in enumerate(folderNames):
        _,trackingFolder = AZU.get_analysis_folders(folder)
        
        # grab tracking files
        trackingSubFiles = glob.glob(trackingFolder + r'\*.npz')
        
        # grab aviFiles
        aviSubFiles=glob.glob(folder + r'\*.avi')
        
        if len(aviSubFiles)<len(trackingSubFiles):
            sys.exit('Could not find movie files for all tracking files...exiting')
        elif len(aviSubFiles)>len(trackingSubFiles):
            sys.exit('Could not find tracking files for all avi files...exiting')
            
        # add to overall lists (one by one)
        for i,s in enumerate(trackingSubFiles):
            trackingNameList.append(s)
            aviNameList.append(aviSubFiles[i])
        
elif(len(folderListFile)==0 and len(trackingSHFolder)!=0): # then we are dealing with a folder of shortcuts
        opt=0
        # cycle through the shortcuts and compile a list of targets
        shFiles=glob.glob(trackingSHFolder+'\*.lnk')
        for i in range(len(shFiles)):
            ret,path=AZU.findShortcutTarget(shFiles[i])
            if(ret==0):
                wDir,_,f=path.rsplit(sep='\\',maxsplit=2)
                trackingFolder=wDir + '\\Tracking\\'
                f=f[0:-13]
                trackingNameList.append(glob.glob(trackingFolder + r'\\*' + f + '*.npz'))
                aviNameList.append(glob.glob(wDir + r'\\*' + f + '*.avi'))
            else:
                print('Broken Link detected for' + f)
elif((len(folderListFile)==0 and len(trackingFolder)==0) or opt==-1):
    sys.exit('No tracking shortcut folder or FolderlistFile provided...exiting')
           
# remove any that we don't have both the avi and the trackingfile for            
tBool=[]

for i,s in enumerate(trackingNameList):
    tBool.append(s!=[] and aviNameList[i]!=[])
    
trackinglList = [i for indx,i in enumerate(trackingNameList) if tBool[indx]]
avilList = [i for indx,i in enumerate(aviNameList) if tBool[indx]]

trackingList=[]
aviList=[]

for i,_ in enumerate(trackinglList):
    if opt==0:  
        trackingList.append(trackinglList[i][0])
        aviList.append(avilList[i][0])
    elif opt==1:
        trackingList.append(trackinglList[i])
        aviList.append(avilList[i])
        
# check nothing went dramatically wrong with these steps
if len(trackingList)!=len(aviList):
    sys.exit('Something went very wrong! The aviList and trackingList are different lengths... exiting')
    
# Loop through files tracking data             
for i,trackingFile in enumerate(trackingList):
    vid=cv2.VideoCapture(aviList[i])
    print('Performing spatial analysis on ' + trackingFile)
    wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
    fx,fy,bx,by,ex,ey,area,ort,motion=AZU.grabTrackingFromFile(trackingFile)
    
    # if sameROIs is False, this module displays frame 10 of the selected movie and gives you three seconds to press escape and define new ROIs. Otherwise, the previous ROIs will be used
    if sameROIs==False:
        img = AZU.grabFrame(vid,10)
        cv2.namedWindow('Check', flags=cv2.WND_PROP_FULLSCREEN)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,0,255)
        lineType               = 2

        cv2.putText(img,'You have 3 seconds to press "escape" if you want to define new ROIs', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('Check', img)
        if cv2.waitKey(3000) == 27: # wait for "escape" for 3 sec
            ROI_masks=[]
        cv2.destroyWindow('Check')
        
    if len(ROI_masks)!=numROIs:
        print('Define your ROIS...')   
            
        # loop through ROIs, and define each one by polygon selection performed on a pop-up of the background image
        ## N.B SHOULD DO THIS BY PLOTTING ONLY ONE ROI IN THE CENTRE THEN DRAWING LINES OUT THE EDGES OF THE IMAGE FOR THE REMAINING ROIS
        ROI_poly=[]
        ROI_masks=[]
        while len(ROI_poly)!=numROIs:
            for ROIName in ROINames:
                polydata,mask=AZR.drawPoly(vid,ROIName) # ROI GUI
                ROI_poly.append(polydata.points)
                ROI_masks.append(mask)
    
    # we now have 9 (maybe) ROIs in the form of polygon points and binary masks. We want to save these in the dictionary format but we'll worry about that later
    # Now we want to define which bits of the movie the fish is in on a frame by frame basis. 
    
    # create empty arrays to draw where the fish is and to tag frames for each ROI
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ROI_Tag=np.zeros(len(bx))
    ROI_Tag[:]=-1
    framesInROI=np.zeros(numROIs)
    timeInROI_PC=np.zeros(numROIs)
    numFrames=len(bx)
    
    # tag frames with ROI_Tag
    for f in range(numFrames):
        if(f==0):print('Checking ROIs for each frame...')
        if(f==np.floor(numFrames/2)): print('Halfway there...')
        if(f==numFrames-1):print('Done')
        # check point is inside an ROI each frame
        for i,mas in enumerate(ROI_masks):
 
            bxx=bx[f]
            byy=by[f]
            # check if coords are inside ROI
            if bxx > mas.shape[0]-1:bxx=mas.shape[0]-1
            if byy > mas.shape[1]-1:byy=mas.shape[1]-1
            if(mas[int(np.floor(bxx)),int(np.floor(byy))]==1):
                ROI_Tag[f]=i
                break
        if f>0 and ROI_Tag[f]==-1:  # if no ROI is found, assume fish is in the same ROI as previous frame
            ROI_Tag[f]=ROI_Tag[f-1]
            
    # now grab the boutStarts from file (non ROI dictionary)
    AnalysisFolder=wDir+'\\Analysis\\'
    thisFishDictName=AnalysisFolder+name+'_ANALYSIS.npy'
    dic=np.load(thisFishDictName,allow_pickle=True).item()
    starts = dic['data']['boutStarts']
    
    # tag bouts with ROI_tag
    ROI_boutTag=np.zeros(len(starts))
    ROI_boutTag[:]=-1
    for i,boutStart in enumerate(starts):
        if(i==0):print('Checking ROIs for each bout...')
        if(i==np.floor(len(starts)/2)): print('Halfway there...')
        if(i==(len(starts)-1)):print('Done')
        
        while ROI_Tag[boutStart]==-1: # if the start frame for this bout happens to be -1 (could not determine ROI for this frame) then cycle along until you find one
            boutStart+=1
            
        ROI_boutTag[i]=ROI_Tag[boutStart]
        
    ## Now I want to section out the tracking data for all chunks of each ROI and compute all our things for them. Easiest way to do this with existing code is to write seperate tracking files for each segment that we then feed into the analysis. 
    # make a bunch of empty lists to keep track of data. We can place these one by one, and is a bit clunky to do it with lists like this but it's easier to code basically
    BPSS=[]
    allBoutsS=[]
    allBoutsOrtS=[]
    allBoutsDistS=[]
    LTurnPCS=[]
    boutAnglesS=[]
    boutSeqS=[]
    seqProb1S=[]
    seqProb2S=[]
    avgBoutS=[]
    avgBoutSES=[]
    biasLeftBoutS=[]
    avgAngVelocityBoutS=[]
    boutAmpsS=[]
    boutDistS=[]
    seg_data=[]
    distPerFrameS=[]
    avgVelocityS=[]
    cumDistS=[]
    seqProbs2_VS=[]
    seqProbs2_PS=[]
    seqProbs2_ZS=[]
    
    data=dic['data']
    dist = data['distPerFrame']
    # Now we have tagged every frame with the ROI the fish is in for each point. First let's compute time spent in each ROI 
    for r in range(numROIs):
        framesInROI[r]=np.sum(np.asarray(ROI_Tag==r))
        timeInROI_PC[r]=np.format_float_positional((framesInROI[r]/numFrames),precision=4, unique=False, fractional=False, trim='k')

#        print('Fish spent ' + str(timeInROI_PC[r]*100) + '% of the movie in the ' + ROINames[r])    
        
        # now section out dist trace by ROI
        # check nothing is empty (if fish never went to this ROI). Replace with zero if it is
        if len(dist)!=0:
            distPerFrame = dist[ROI_Tag==r] 
            if len(distPerFrame)==0:
                distPerFrame=[0]
        else: 
            distPerFrame=[0]
        if len(bx)!=0:
            bx_s=bx[ROI_Tag==r] 
            if len(bx_s)==0:bx_s=[0]
        else: bx_s=[0]
        if len(by)!=0:
            by_s=by[ROI_Tag==r] 
            if len(by_s)==0:by_s=[0]
        else: by_s=[0]
        if len(ort)!=0:
            ort_s=ort[ROI_Tag==r] 
            if len(ort_s)==0:ort_s=[0]
        else: ort_s=[0]
        bx_smm,by_smm = AZU.convertToMm(bx_s,by_s) 
        distPerSec=distPerFrame*FPS
        
        # recompute things for this ROI
        ##### OVERALL METRICS #####
        ### distPerFrame ###
        distPerFrameS.append(distPerFrame)
        if len(distPerFrame)>1:
            ### cumDist ###
            cumDist=AZU.accumulate(distPerFrame)
            cumDistS.append(cumDist)
            ### avgVelocity ###
            avgVelocityS.append(cumDist[-1] /(len(cumDist)/FPS))
        else: 
            ### cumDist ###
            cumDistS.append(-1)
            ### avgVelocity ###
            avgVelocityS.append(-1)
            
        ##### BOUT METRICS #####
        ### Count the number of bouts in this ROI ###
        keep=ROI_boutTag==r
        boutsThisROI=np.sum(keep)
        
        # if there are bouts in this ROI...
        if boutsThisROI!=0:
            ### BPS ###
            BPSS.append(boutsThisROI/(framesInROI[r]/FPS))  
            ### boutAmps ###
            ROIamps=(data['boutAmps'])[keep]
            boutAmpsS.append(ROIamps)
            ### boutDists ###
            ROIdists=(data['boutDists'])[keep]
            boutDistS.append(ROIdists)
            ### allBoutsDist ###
            allBoutsDistS.append((data['allBoutsDist'])[keep])
            ### allBoutsOrts ###
            allBoutsOrtS.append((data['allBoutsOrts'])[keep])
            ### allBouts ###
            ROIBouts=(data['allBouts'])[keep]
            allBoutsS.append(ROIBouts)
            ### avgBout mean and SE ###
            avgBoutS.append(np.mean(ROIBouts,0))
            avgBoutSES.append(np.std(ROIBouts,0)/np.sqrt(len(ROIdists)))
            ### boutAngles ###
            ROIangles=(data['boutAngles'])[keep]
            boutAnglesS.append(ROIangles)
            ### avgAngularVelocity ###
            avgAngVelocityBoutS.append(np.mean(np.abs(ROIangles)))
            ### biasLeftBout ###
            biasLeftBoutS.append((np.sum(ROIangles))/(np.sum(np.abs(ROIangles))))
            ### boutSeq ###
            fullSeq=data['boutSeq']
            ROISeq=fullSeq[keep]
            boutSeqS.append(ROISeq)
            ### seqProbs_1 ###
            ## N.B. dictionaries cannot be put together in this loop as if the first ROI has no bouts then comb1 and comb2 will not be defined yet
            if len(ROISeq)>6: 
                comb1,v1=AZP.probSeq1(ROISeq)
                comb2,_,v2,z,p=AZP.probSeq2_ROI(data['boutSeq'],ROISeq)
                
            else: v1=-1 ; v2=-1 ; z=-1 ; p=-1
            
            seqProb1S.append(v1)
            ### seqProbs_2 ###
                ### Probabilities ###
            seqProbs2_VS.append(v2)
                ### ZScores ###
            seqProbs2_ZS.append(z)
                ### pvalues ###
            seqProbs2_PS.append(p)
            
        else:
            ### BPS ###
            BPSS.append(0)
            ### boutAmps ###
            boutAmpsS.append(-1)
            ### boutDists ###
            boutDistS.append(-1)
            ### allBouts ###
            allBoutsS.append(-1)
            ### allBoutsDist ###
            allBoutsDistS.append(-1)
            ### allBoutsOrts ###
            allBoutsOrtS.append(-1)
            ### avgBout mean and SE ###
                ## Mean ##
            avgBoutS.append(-1)
                ## SEM ##
            avgBoutSES.append(-1)
            ### boutAngles ###
            boutAnglesS.append(-1)
            ### avgAngularVelocity ###
            avgAngVelocityBoutS.append(-1)
            ### biasLeftBout ###
            biasLeftBoutS.append(-1)
            ### seqProbs1 ###
            seqProb1S.append(-1)
            ### seqProbs_2 ###
                ## Probabilities ##
            seqProbs2_VS.append(-1)
                ## ZScores ##
            seqProbs2_ZS.append(-1)
                ## pvalues ##
            seqProbs2_PS.append(-1)
        
#        boutStarts=[]
#        LTurnPC=[]
#        if len(distPerFrame)>1:
#            cumDist=AZU.accumulate(distPerFrame)
#            avgVelocity=cumDist[-1] /(len(cumDist)/FPS)
#            idF=AnalysisFolder+'Figures\\ROIs\\'
#            AZU.cycleMkDir(idF)
##            ret, RTurns,LTurns,FSwims,BPS, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC,boutStarts,_,_,_ = AZS.extractBouts(bx_smm,by_smm,ort_s,distPerSec,name=name+'_ROI_' + ROINames[r], savepath=idF,plot=False)        
#            
#            
#        else:
#            cumDist=[0]
#            avgVelocity=0
#        
#        if(ret==-1) or len(boutStarts)<=1: # if no or only one bout detected, set everything to -1
#            
##            BPSS.append(-1)
#            avgBoutS.append(-1)
#            avgBoutSES.append(-1)
#            biasLeftBoutS.append(-1) 
#            avgAngVelocityBoutS.append(-1)
#            boutAmpsS.append(-1)
#            boutDistS.append(-1)
#            allBoutsS.append(-1)
#            allBoutsOrtS.append(-1)
#            LTurnPCS.append(-1)
#            boutAnglesS.append(-1)
#            boutSeqS.append(-1)
#            distPerFrameS.append(-1)
#            avgVelocityS.append(-1)
#            cumDistS.append(-1)
#            seqProbS.append(-1)
#        else:
#            # check bouts for silly measures
#            amm=np.where((np.max(allBoutsDist,axis=1))>50)
#            bmm=np.where((np.max(allBoutsDist,axis=1))<3) # exclude any that do not travel faster than 3mm/s
#            keep=np.ones(len(boutAngles))
#            keep[amm]=0
#            keep[bmm]=0
#            keep=keep>0
##    
#            RTurns=RTurns[keep]
#            LTurns=LTurns[keep]
#            FSwims=FSwims[keep]
#            allBouts = allBouts[keep]
#            allBoutsDist = allBoutsDist[keep]
#            allBoutsOrt = allBoutsOrt[keep]
#            boutAngles = boutAngles[keep]
#            boutStarts = boutStarts[keep]
#            BPSS.append(BPS)
#            allBoutsS.append(allBouts)
#            allBoutsOrtS.append(allBoutsOrt)
#            LTurnPCS.append(LturnPC)
#            boutAnglesS.append(boutAngles)
#            boutSeqS.append(AZP.angleToSeq(boutAngles))
#            if len(boutAngles)>3:
#                comb,ss=AZP.probSeq1(boutSeqS[r])
#            else: ss=-1
#            seqProbS.append(ss)
#            avgBoutS.append(np.mean(allBouts,0))
#            avgBoutSES.append(np.std(allBouts,0)) #/np.sqrt(len(boutAngles)))
#            biasLeftBoutS.append((np.sum(boutAngles))/(np.sum(np.abs(boutAngles)))) # positive is bias for left, negative bias for right
#            avgAngVelocityBoutS.append(np.mean(np.abs(boutAngles)))
#            distPerFrameS.append(distPerFrame)
#            avgVelocityS.append(avgVelocity)
#            cumDistS.append(cumDist)
#            
            # Compute bout amplitudes from all bouts peak
#            boutAmpsS.append(AZA.findBoutMax(allBouts))
#            OR
            # Compute boutAmps from integral of distance travelled during that bout
#            print('There are ' + str(len(boutStarts)) + ' bouts in this ROI')
#            print('First bout starts at frame ' + str(boutStarts[0]))
#            boutDistS.append(AZA.findBoutArea(allBoutsDist,FPS))
        
        thisMask=ROI_masks[r]
        thisprob=seqProbS[r]
        seg_data.append({'ROIName'              :   ROINames[r],
                         'BPS'                  :   BPSS[r],
                         'avgVelocity'          :   avgVelocityS[r],
                         'avgAngVelocityBout'   :   avgAngVelocityBoutS[r],
                         'biasLeftBout'         :   biasLeftBoutS[r],
                         'LTurnPC'              :   LTurnPCS[r],
                         'distPerFrame'         :   distPerFrameS[r],
                         'cumDist'              :   cumDistS[r],
                         'avgBout'              :   {'Mean'             :   avgBoutS[r],
                                                     'SE'               :   avgBoutSES[r],
                                                     },
                         'boutAmps'             :   boutAmpsS[r],
                         'boutDists'            :   boutDistS[r],
                         'boutAngles'           :   boutAnglesS[r],
                         'allBouts'             :   allBoutsS[r],
                         'boutOrts'             :   allBoutsOrtS[r],
                         'ROIMask'              :   thisMask,
                         'PCTimeSpent'          :   timeInROI_PC[r],            
                         'boutSeq'              :   boutSeqS[r],
                         'seqProbs'             :   {'comb'  :   comb,
                                                     'prob'  :   thisprob,
                                                     },
                         })
                         
        # save the dictionary
    dic['ROIs']= seg_data
    outName=AnalysisFolder+name+'_ANALYSIS_ROIs.npy'
    print('Saving ROI data at ' + outName)
    np.save(outName,dic)
    
        
        
       
       
        