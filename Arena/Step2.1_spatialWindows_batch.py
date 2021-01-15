# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:46:22 2020

@author: thoma
"""
import os
# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
import sys
sys.path.append(lib_path)
import numpy as np
import AZ_utilities as AZU
import glob
import AZ_ROITools as AZR
import AZ_streakProb as AZP
import cv2

folderListFile=[]
trackingFolder=[]
trackingSHFolder=[]

###############################################################################
# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile = r'D:\Movies\FolderLists\AllCtrl.txt'

# OR
trackingSHDir = r'D:\\Movies\\GroupedData\\Groups\\'
trackingSHFolders =[ trackingSHDir + r'EC_B0',trackingSHDir + r'EC_M0',trackingSHDir + r'EA_B0',trackingSHDir + r'EA_M0'] 
inSuffs=['EC_B0','EC_M0','EA_B0','EA_M0']
outSuffs=['EC_B0','EC_M0','EA_B0','EA_M0']
AnalysisFolderRoot=r'D:\\Analysis'
# Set Flags
createSpatialFigures=True
keepSpatialFigures=False
sameROIs = False
omitForward=False

# Specify start and end frame that is included from tracking data
sf=0*60*120
ef=-1

# Specify the names of all ROIs you have or want to define
#ROINames=[]
CSName=['Centre','Surround']
M0Name=['Top Left','Top','Top Right','Middle Left','Central Chamber', 'Middle Right', 'Bottom Left','Bottom','Bottom Right']
ROINamesS=[CSName,M0Name,CSName,M0Name]
#ROINames=['Centre']
#ROINames=['Top Left','Top','Top Right','Middle Left','Central Chamber', 'Middle Right', 'Bottom Left','Bottom','Bottom Right']
#ROI_masks=np.load('C:/Users/thoma/OneDrive/Documents/GitHub/Arena_Zebrafish/Arena/M0_ROIMasks.npy')
CSMask='C:/Users/thoma/OneDrive/Documents/GitHub/Arena_Zebrafish/Arena/B0_CentreSurround.npy'
M0Mask='C:/Users/thoma/OneDrive/Documents/GitHub/Arena_Zebrafish/Arena/M0_ROIMask.npy'
ROI_masksS=[CSMask,M0Mask,CSMask,M0Mask]
ROI_masks=np.load('C:/Users/thoma/OneDrive/Documents/GitHub/Arena_Zebrafish/Arena/B0_CentreSurround.npy')
# other group parameters
FPS=120

######################## BEGIN BIG LOOP #######################################
for bigLoop,trackingSHFolder in enumerate(trackingSHFolders):
    inSuff=inSuffs[bigLoop]
    outSuff=outSuffs[bigLoop]
    ROINames=ROINamesS[bigLoop]
    ROI_masks=np.load(ROI_masksS[bigLoop])
        
    
    if omitForward:
        inSuff=inSuff+'_FO'
        outSuff=outSuff+'_FO'
        
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
            _,templateFolder,trackingFolder = AZU.get_analysis_folders(folder)
            
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
                    templateFolder=wDir + '\\Templates\\'
                     
                    f=f[0:-13]
                    trackingNameList.append(glob.glob(trackingFolder + r'\\*' + f + '*.npz'))
                    aviCheck=glob.glob(templateFolder + r'\\*' + f + '*.avi')
                    if aviCheck==[]: aviCheck=glob.glob(wDir + r'\\*' + f + '*.avi')
                    aviNameList.append(aviCheck)
                else:
                    print('Broken Link detected for' + f)
    elif((len(folderListFile)==0 and len(trackingFolder)==0) or opt==-1):
        sys.exit('No tracking shortcut folder or FolderlistFile provided...exiting')
               
    # remove any that we don't have both the avi and the trackingfile for            
    tBool=[]
    
    for i,s in enumerate(trackingNameList):
        tBool.append(s!=[] and aviNameList[i]!=[])
    #    tBool.append(True)
        
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
    
    # check through to make sure all files exist
    print('Checking files exist...')
    for i,trackingFile in enumerate(trackingFiles):
        if os.path.exists(trackingFile)==False:
            print('Error, ' + trackingFile + ' does not exist... removing from list')
            trackingFiles.remove(trackingFile)
            missingFiles.append(trackingFile)
            
    # Loop through files tracking data             
    for i,trackingFile in enumerate(trackingList):
        avi=aviList[i]
        print('Performing spatial analysis on ' + trackingFile)
        wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
        fx,fy,bx,by,ex,ey,area,ort,motion=AZU.grabTrackingFromFile(trackingFile)
        if(ef>len(fx)) : ef = -1
        # check that the analysis includes all the tracking and crop if needed. This might be needd if you are, for example, looking at a subsection of a movie's tracking data
        fx=fx[sf:ef]
        fy=fy[sf:ef]
        bx=bx[sf:ef]
        by=by[sf:ef]
        ex=ex[sf:ef]
        ey=ey[sf:ef]
        area=area[sf:ef]
        ort=ort[sf:ef]
    
        # if sameROIs is False, this module displays template frame created during tracking of the selected movie and gives you three seconds to press escape and define new ROIs. Otherwise, the previous ROIs will be used
        if sameROIs==False:
            img = AZU.grabFrame(avi,0)
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
                    polydata,mask=AZR.drawPoly(avi,ROIName) # ROI GUI
                    ROI_poly.append(polydata.points)
                    ROI_masks.append(mask)
        
        # we now have 9 (maybe) ROIs in the form of polygon points and binary masks. We want to save these in the dictionary format but we'll worry about that later
        # Now we want to define which bits of the movie the fish is in on a frame by frame basis. 
        
        # create empty arrays to draw where the fish is and to tag frames for each ROI
        vid=cv2.VideoCapture(aviList[i])
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid.release()
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
        if ROI_masks==[]:ROI_Tag[:]=0
                    
        # now grab the boutStarts from file (non ROI dictionary)
        AnalysisFolder=AnalysisFolderRoot + inSuff + '\\Dictionaries\\'
        thisFishDictName=AnalysisFolder+name+'_ANALYSIS_' + inSuff + '.npy'
        dic=np.load(thisFishDictName,allow_pickle=True).item()
        starts = dic['data']['boutStarts']
        
        # tag bouts with ROI_tag
        ROI_boutTag=np.zeros(len(starts))
        ROI_boutTag[:]=-1
        for i,boutStart in enumerate(starts):
            if(i==0):print('Checking ROIs for each bout...')
            if(i==np.floor(len(starts)/2)): print('Halfway there...')
            if(i==(len(starts)-1)):print('Done')
            
            while ROI_Tag[boutStart]==-1 and boutStart<numFrames: # if the start frame for this bout happens to be -1 (could not determine ROI for this frame) then cycle along until you find one
                boutStart+=1
            
            ROI_boutTag[i]=ROI_Tag[boutStart]
            
        ## Now I want to section out the tracking data for all chunks of each ROI and compute all our things for them. Easiest way to do this with existing code is to write seperate tracking files for each segment that we then feed into the analysis. 
        # make a bunch of empty lists to keep track of data. We can place these one by one, and is a bit clunky to do it with lists like this but it's easier to code basically
        BPSS=[]
        allBoutsS=[]
        allBoutsOrtS=[]
        allBoutsDistS=[]
        LturnPCS=[]
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
    #    while len(dist)>len(ROI_Tag): ROI_Tag.append(-1)
    #    while len(ROI_Tag)>len(dist): ROI_Tag=ROI_Tag[:-1]
        ##### START OF ROI LOOP 1 #####
        for r in range(numROIs):
            if(r==0):print('Looping through ROIs')
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
            
            if len(distPerFrame)>1:
                ### distPerFrame ###
                distPerFrameS.append(distPerFrame)
                ### cumDist ###
                cumDist=AZU.accumulate(distPerFrame)
                cumDistS.append(cumDist)
                ### avgVelocity ###
                avgVelocityS.append(cumDist[-1] /(len(cumDist)/FPS))
            else: 
                ### distPerFrame ###
                distPerFrameS.append(-1)
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
                if omitForward:ROISeq=AZP.angleToSeq_LR(ROIangles)
                else:ROISeq=AZP.angleToSeq(ROIangles)
                boutSeqS.append(ROISeq)
                
                ### seqProbs_1 ###
                ## N.B. dictionaries cannot be put together in this loop as if the first ROI has no bouts then comb1 and comb2 will not be defined yet
                if len(ROISeq)>5: 
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
                ### LTurnPC ###
                numL=0
                numT=0
                for k in ROISeq:
                    if k=='L':
                        numL+=1
                        numT+=1
                    elif k=='R':
                        numT+=1
    
                if numL!=0 and numT!=0:
                    LturnPCS.append((numL/numT)*100)
                else: 
                    LturnPCS.append(-1)
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
                ### boutSeq ###
                boutSeqS.append(-1)
                ### seqProbs1 ###
                seqProb1S.append(-1)
                ### seqProbs_2 ###
                    ## Probabilities ##
                seqProbs2_VS.append(-1)
                    ## ZScores ##
                seqProbs2_ZS.append(-1)
                    ## pvalues ##
                seqProbs2_PS.append(-1)
                ### LTurnPC ###
                LturnPCS.append(-1)
                boutSeqS.append(-1)
        ##### END OF ROI LOOP 1 #####
        
        seg_data=[]          
        ##### START OF ROI LOOP 2 #####
        for r in range(numROIs):
            seg_data.append({'ROIName'              :   ROINames[r],
                             'BPS'                  :   BPSS[r],
                             'avgVelocity'          :   avgVelocityS[r],
                             'avgAngVelocityBout'   :   avgAngVelocityBoutS[r],
                             'biasLeftBout'         :   biasLeftBoutS[r],
                             'LTurnPC'              :   LturnPCS[r],
                             'distPerFrame'         :   distPerFrameS[r],
                             'cumDist'              :   cumDistS[r],
                             'avgBout'              :   {'Mean'             :   avgBoutS[r],
                                                         'SE'               :   avgBoutSES[r],
                                                         },
                             'boutAmps'             :   boutAmpsS[r],
                             'boutDists'            :   boutDistS[r],
                             'boutAngles'           :   boutAnglesS[r],
                             'allBouts'             :   allBoutsS[r],
                             'allBoutsDist'         :   allBoutsDistS[r],
                             'boutOrts'             :   allBoutsOrtS[r],
                             'ROIMask'              :   ROI_masks[r],
                             'PCTimeSpent'          :   timeInROI_PC[r],            
                             'boutSeq'              :   boutSeqS[r],
                             'seqProbs1'             :   {'comb'    :   comb1,
                                                         'prob'     :   seqProb1S[r],
                                                         },
                             'seqProbs2'            :   {'comb'     :   comb2,
                                                         'prob'     :   seqProbs2_VS[r],
                                                         'pvalues'  :   seqProbs2_PS[r],
                                                         'zscores'  :   seqProbs2_ZS[r],
                                                         },
                             })
                             
            # save the dictionary
        ############ END OF ROI LOOP 2 ########################
        dic['ROIs']= seg_data
        outName=AnalysisFolder+name+'_ANALYSIS_ROIs' + outSuff + '.npy'
        print('Saving ROI data at ' + outName)
        np.save(outName,dic)
    ############### END OF FILE LOOP ####################
############## END BIG LOOP ####################
            
            
           
       
        