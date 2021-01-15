# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:04:43 2020

@author: thoma
"""

# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'

import sys
sys.path.append(lib_path)
import numpy as np
import AZ_utilities as AZU
import AZ_figures as AZF 


folderListFile=[]
trackingFolder=[]
###############################################################################
# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile = r'D:\Movies\FolderLists\AllCtrl.txt'

# OR
trackingSHDir = r'D:\\Movies\\GroupedData\\Groups\\'
trackingSHFolders =[ trackingSHDir + r'EC_B0',trackingSHDir + r'EC_M0',trackingSHDir + r'EA_B0',trackingSHDir + r'EA_M0'] 
inSuffs=['EC_B0','EC_M0','EA_B0','EA_M0']
outSuffs=inSuffs
anSuffs=inSuffs
AnalysisFolderRoot=r'D:\\Analysis'
groupNames=inSuffs

# Set Flags
createGroupFigures=True
keepGroupFigures=True
report=True
omitForward=False
FPS=120

# Make Dictionary and Figure Folders and define output paths
outPathD='D:\\Movies\\GroupedData\\Dictionaries\\'
outPathF='D:\\Movies\\GroupedData\\Figures_210114\\'
################# START BIG LOOP ##############################################
for bigLoop,trackingSHFolder in enumerate(trackingSHFolders):
    
    inSuff=inSuffs[bigLoop]
    outSuff=outSuffs[bigLoop]
    anSuff=anSuffs[bigLoop]
    groupName=groupNames[bigLoop]



    groupName = groupName + outSuff
    if omitForward:
        inSuff=inSuff+'_FO'
        outSuff=outSuff+'_FO'
        groupName=groupName+'_FO'
    
    trackingFiles=[]
    ROIdictNameList=[]
    dictList=[]
    ROIdictList=[]
    
    ## find dictionaries according to tracking or file folderlist
    #if(len(folderListFile)!=0 and len(trackingFolder)==0): # then we are dealing with a folderList rather than a folder of shortcuts
    #    dictNameList=AZU.getDictsFromFolderList(folderListFile)    
    #        
    #elif(len(folderListFile)==0 and len(trackingFolder)!=0): # then we are dealing with a folder of shortcuts
    #    dictNameList=AZU.getDictsFromTrackingFolderROI(trackingFolder,anSuff=anSuff,suff=inSuff)    
    #        
    #elif(len(folderListFile)==0 and len(trackingFolder)==0):
    #            sys.exit('No tracking folder or FolderlistFile provided')
    
    # Check here if te dictNameList is empty. If it is, check the Root Data folder
    #if len(dictNameList)==0:
    dictNameList=AZU.getDictsFromRootFolderROI(trackingFolder,anSuff=anSuff,suff=inSuff) 
     ########################################################################
               
    AZU.cycleMkDir(outPathD)
    oF=outPathF+groupName + '\\'
    AZU.cycleMkDir(oF)
    
    dictPath_out=outPathD + groupName + '.npy'
    numFiles=len(dictNameList)
    
    indHeatMaps=np.zeros((10,10,numFiles))# heatmap
    group_BPS=[]
    group_avgVelocity=[]
    group_avgAngVelocityBout=[]
    group_biasLeftBout=[]
    group_LTurnPC=[]
    group_cumDist=[]
    group_avgBout=[]
    group_heatmap=[]
    group_avgBoutAmps=[]
    group_avgBoutDists=[]
    group_avgBoutAngles=[]
    group_avgBoutOrts=[]
    
    group_AvgAmps=[]
    group_AvgDists=[]
    group_AvgOrts=[]
    
    group_allBoutAmps=[]
    group_allBoutDists=[]
    group_allBoutAngles=[]
    
    group_seqProbS1=[]
    group_seqProbS2_Z=[]
    
    # load the first dictionary to count ROIs
    
    firstDic=np.load(dictNameList[0],allow_pickle=True).item()
    numROIs=len(firstDic['ROIs'])
    comb1=firstDic['data']['seqProbs1']['comb']
    comb2=firstDic['data']['seqProbs2']['comb']
    combROI1=firstDic['ROIs'][0]['seqProbs1']['comb']
    combROI2=firstDic['ROIs'][0]['seqProbs2']['comb']
    numClass2=len(comb2)
    numClass1=len(comb1)
    numClassROI2=len(combROI2)
    numClassROI1=len(combROI1)
    dim=[numFiles,numROIs]
    ROI_BPS=np.zeros(dim)
    ROI_avgVelocity=np.zeros(dim)
    ROI_avgAngVelocityBout=np.zeros(dim)
    ROI_biasLeftBout=np.zeros(dim)
    ROI_LTurnPC=np.zeros(dim)
    ROI_AvgBoutAmps=np.zeros(dim)
    ROI_AvgBoutDists=np.zeros(dim)
    ROI_AvgBoutAngles=np.zeros(dim)
    ROI_seqProb2_VS=np.zeros((numFiles,numClassROI2,numROIs))
    ROI_seqProb1S=np.zeros((numFiles,numClassROI1,numROIs))
    ROI_seqProb2_PS=np.zeros((numFiles,numClassROI2,numROIs))
    ROI_seqProb2_ZS=np.zeros((numFiles,numClassROI2,numROIs))
    
    
    ROI_PCTimeSpent=np.zeros(dim)
    
    ROINames=[]
    ROIMasks=[]
    ROI_BoutDists=[]
    ROI_BoutAngles=[]
    ROI_BoutAmps=[]
    ROI_BoutSeq=[]
    
    
    
    for i,f in enumerate(dictNameList):
        dic=np.load(f,allow_pickle=True).item()
        dictList.append(dic)
        data=dic['data']
        ROIs=dic['ROIs']
        
        ###################################################################################################       
        # grab ROINames and masks on the first loop
        if i==0:
            for j in range(numROIs):
                ROINames.append(ROIs[j]['ROIName'])
                ROIMasks.append(ROIs[j]['ROIMask'])
            if report:print('Checking Files for ROI consistency')
            for fff in dictNameList:
                ddd=np.load(fff,allow_pickle=True).item()
                RRR=ddd['ROIs']
                ROINamesCheck=[]
                
                for k in range(len(RRR)):
                    ROINamesCheck.append(RRR[k]['ROIName'])
                    
                if ROINames!=ROINamesCheck:sys.exit('Error: ROIs are not consistent across the selected group!')
            print('ROIs are consistent')
            
        group_seqProbS1.append(data['seqProbs1']['prob'])
        group_seqProbS2_Z.append(data['seqProbs2']['zscores'])
        
        group_BPS.append(data['BPS'])
        group_avgVelocity.append(data['avgVelocity'])
        group_avgAngVelocityBout.append(data['avgAngVelocityBout'])
        group_biasLeftBout.append(data['biasLeftBout']) 
        group_LTurnPC.append(data['LTurnPC'])
        group_avgBoutAmps.append(np.mean(data['boutAmps']))
        group_avgBoutDists.append(np.mean(data['boutDists']))
        group_avgBoutAngles.append(np.mean(data['boutAngles']))
        group_avgBoutOrts.append(np.mean(data['allBoutsOrts']))
        group_cumDist.append(data['cumDist'])
        group_heatmap.append(data['heatmap'])
        group_avgBout.append(data['avgBout']['Mean'])
        
        group_allBoutAmps.append(data['boutAmps'])
        group_allBoutDists.append(data['boutDists'])
        group_allBoutAngles.append(data['boutAngles'])
        
        # take an average for each fish
        for u,av in enumerate(group_avgBoutAmps):
            group_AvgAmps.append(np.mean(av))
            group_AvgOrts.append(np.mean(group_avgBoutOrts[u]))
            group_AvgDists.append(np.mean(group_avgBoutDists[u]))
            
        mL=(60*60*FPS)
        lL=(4*60*FPS)
        shortest=mL
        for cu in group_cumDist:
           if((len(cu)<shortest) and (len(cu)>lL)):shortest=len(cu)
               
        group_cumDistcrop=[]
        for cu in group_cumDist:
            if(len(cu)>= shortest):group_cumDistcrop.append(cu[0:shortest]) # if this is not as long as the minimum then remove (only for the avg group figure purpose)
            
        ###################################################################################################  
        
        for ROIi in range(numROIs):        
            ROI_BPS[i,ROIi]                =   ROIs[ROIi]['BPS']
            ROI_avgVelocity[i,ROIi]        =   ROIs[ROIi]['avgVelocity']
            ROI_avgAngVelocityBout[i,ROIi] =   ROIs[ROIi]['avgAngVelocityBout']
            ROI_biasLeftBout[i,ROIi]       =   ROIs[ROIi]['biasLeftBout']
            ROI_LTurnPC[i,ROIi]            =   ROIs[ROIi]['LTurnPC']
            ROI_AvgBoutAmps[i,ROIi]        =   np.mean(ROIs[ROIi]['boutAmps'])
            ROI_AvgBoutDists[i,ROIi]       =   np.mean(ROIs[ROIi]['boutDists'])
            ROI_AvgBoutAngles[i,ROIi]      =   np.mean(ROIs[ROIi]['boutAngles'])
            ROI_PCTimeSpent[i,ROIi]        =   ROIs[ROIi]['PCTimeSpent']
            ROI_seqProb1S[i,:,ROIi]           =(ROIs[ROIi]['seqProbs1']['prob'])
            ROI_seqProb2_VS[i,:,ROIi]         =(ROIs[ROIi]['seqProbs2']['prob'])
            ROI_seqProb2_ZS[i,:,ROIi]         =(ROIs[ROIi]['seqProbs2']['zscores'])
            ROI_seqProb2_PS[i,:,ROIi]         =(ROIs[ROIi]['seqProbs2']['pvalues'])
            
            ROI_BoutAngles.append(ROIs[ROIi]['boutAngles'])
            ROI_BoutAmps.append(ROIs[ROIi]['boutAmps'])
            ROI_BoutDists.append(ROIs[ROIi]['boutDists'])
            ROI_BoutSeq.append(ROIs[ROIi]['boutSeq'])
    #        
    #        if i>0:
    #            
    #            ourAmpList=ROI_BoutAmps[ROIi]
    #            if isinstance(ourAmpList,list)==False and isinstance(ourAmpList,int)==False:
    #                ourAmpList=ourAmpList.tolist()
    #                
    #            thisAmpList=ROIs[ROIi]['boutAmps']
    #            if isinstance(thisAmpList,list)==False and isinstance(thisAmpList,int)==False:
    #                thisAmpList=thisAmpList.tolist()
    #                
    #            ourAmpList.append(thisAmpList)
    #            ROI_BoutAmps[ROIi]=np.asarray(ourAmpList)
    #            
    #            ourAngleList=ROI_BoutAngles[ROIi]
    #            if isinstance(ourAngleList,list)==False and isinstance(ourAngleList,int)==False:
    #                ourAngleList=ourAngleList.tolist()
    #                
    #            thisAngleList=ROIs[ROIi]['boutAngles']
    #            if isinstance(thisAngleList,list)==False and isinstance(thisAngleList,int)==False:
    #                thisAngleList=thisAngleList.tolist()
    #                
    #            ourAngleList.append(thisAngleList)
    #            ROI_BoutAngles[ROIi]=np.asarray(ourAngleList)
    #            
    #            
    #            ourDistList=ROI_BoutDists[ROIi]
    #            if isinstance(ourDistList,list)==False and isinstance(ourDistList,int)==False:
    #                ourDistList=ourDistList.tolist()
    #                
    #            thisDistList=ROIs[ROIi]['boutDists']
    #            if isinstance(thisDistList,list)==False and isinstance(thisDistList,int)==False:
    #                thisDistList=thisDistList.tolist()
    #                
    #            ourDistList.append(thisDistList)
    #            ROI_BoutDists[ROIi]=np.asarray(ourDistList)
    #            
    #            ourAmpList.append(thisAmpList)
    #            ROI_BoutAmps[ROIi]=np.asarray(ourAmpList)
    #            
    #            ourDistList.append(thisDistList)
    #            ROI_BoutDists[ROIi]=np.asarray(ourDistList)
    #            
    #            ROI_BoutAmps[ROIi].extend(ROIs[ROIi]['boutAmps'])
    #            ROI_BoutDists[ROIi].extend(ROIs[ROIi]['boutDists'])
    #            
    #    # flatten lists (instead of lists of lists)    
    #    flat=[item for sublist in ROI_BoutAngles for item in sublist]
    #    ROI_BoutAngles=flat
    #    flat=[item for sublist in ROI_BoutAmps for item in sublist]
    #    ROI_BoutAmps=flat
    #    flat=[item for sublist in ROI_BoutDists for item in sublist]
    #    ROI_BoutDists=flat
               
    ##################################################################################
    # Collate ROI data in dictionary
    seg_data=[]
    for i in range(numROIs):
        thisMask=ROIMasks[i]
        thisprob=ROI_seqProb1S[:,:,i]
        thisV=ROI_seqProb2_VS[:,:,i]
        thisZ=ROI_seqProb2_ZS[:,:,i]
        thisP=ROI_seqProb2_PS[:,:,i]
        seg_data.append({'ROIName'              :   ROINames[i],
                         'BPS'                  :   ROI_BPS[:,i],
                         'avgVelocity'          :   ROI_avgVelocity[:,i],
                         'avgAngVelocityBout'   :   ROI_avgAngVelocityBout[:,i],
                         'biasLeftBout'         :   ROI_biasLeftBout[:,i],
                         'LTurnPC'              :   ROI_LTurnPC[:,i],
                         'avgBoutAmps'          :   ROI_AvgBoutAmps[:,i],
                         'avgBoutDists'         :   ROI_AvgBoutDists[:,i],
                         'ROIMasks'             :   thisMask,
                         'PCTimeSpent'          :   ROI_PCTimeSpent[:,i],
                         'PooledData'           :   {'boutAmps'           :   ROI_BoutAmps[i],
                                                     'boutDists'          :   ROI_BoutDists[i],
                                                     'boutAngles'         :   ROI_BoutAngles[i],
                                                         },
                        'seqProbs1'             :   {'comb'               :   combROI1,
                                                     'prob'               :   thisprob,
                                                    },
                        'seqProbs2'             :   {'comb'               :   combROI2,
                                                     'prob'               :   thisV,
                                                     'pvalues'            :   thisP,
                                                     'zscores'            :   thisZ,
                                                    },
                         })
     ###############################################################################################
     
        
    
    # flatten lists (instead of lists of lists)    
    #flat=[item for sublist in group_allBoutAmps for item in sublist]
    #group_allBoutAmps=flat
    #flat=[item for sublist in group_allBoutDists for item in sublist]
    #group_allBoutDists=flat
    #flat=[item for sublist in group_allBoutAngles for item in sublist]
    #group_allBoutAngles=flat           
    avg_seqProbS1=np.mean(group_seqProbS1,axis=0)
    se_seqProbS1=np.std(group_seqProbS1,axis=0)
    avg_seqProbS2_Z=np.mean(group_seqProbS2_Z,axis=0)
    se_seqProbS2_Z=np.std(group_seqProbS2_Z,axis=0)/(np.sqrt(numFiles))
    
    avgCumDist=(np.mean(group_cumDistcrop,axis=0)) # cumDistAV
    SDCumDist=(np.std(group_cumDistcrop, axis=0)) # cumDistSEM
    avgHeatmap=(np.mean(group_heatmap,axis=0)) # avgHeatmap
    SDHeatmap=(np.std(group_heatmap,axis=0)) # SEMHeatmap
    groupAvgBout=(np.mean(group_avgBout, axis=0)) # avgBoutAV
    groupSEMBout=(np.std(group_avgBout, axis=0)) # avgBoutSEM
    
    # PooledData: all the metrics for which there is a single average value per fish
        
       
    GroupedFish    =    {'Name'                     :   groupName,
                         'Ind_fish'                 :   dictList,
                         'PooledData'               :   {'boutAmps'           :   group_allBoutAmps,
                                                         'boutDists'          :   group_allBoutDists,
                                                         'boutAngles'         :   group_allBoutAngles,
                                                         },
                         'avgData'                  :   {'BPSs'               :   group_BPS,
                                                         'avgVelocity'        :   group_avgVelocity,
                                                         'avgAngVelocityBout' :   group_avgAngVelocityBout,
                                                         'biasLeftBout'       :   group_biasLeftBout,
                                                         'LTurnPC'            :   group_LTurnPC,
                                                         'boutAmps'           :   group_avgBoutAmps,
                                                         'boutDists'          :   group_avgBoutDists,
                                                         'boutOrts'           :   group_avgBoutOrts,
                                                         'cumDists'           :   group_cumDist,
                                                         'avgHeatmaps'        :   group_heatmap,
                                                         'avgBouts'           :   group_avgBout,
                                                         'avg_seqProbS1'      :   group_seqProbS1,
                                                         'avg_seqProbS2_Z'    :   group_seqProbS2_Z
                                                         },
                                                         
                         'Metrics'                  :   {'BPS'                :     {'Mean'    :   np.mean(group_BPS),
                                                                                     'SEM'     :   np.std(group_BPS)},
                                                         'avgVelocity'        :     {'Mean'    :   np.mean(group_avgVelocity),
                                                                                     'SEM'     :   np.std(group_avgVelocity)},
                                                         'avgAngVelocityBout' :     {'Mean'    :   np.mean(group_avgAngVelocityBout),
                                                                                     'SEM'     :   np.std(group_avgAngVelocityBout)},
                                                         'biasLeftBout'       :     {'Mean'    :   np.mean(group_biasLeftBout),
                                                                                     'SEM'     :   np.std(group_biasLeftBout)},
                                                         'LTurnPC'            :     {'Mean'    :   np.mean(group_LTurnPC),
                                                                                     'SEM'     :   np.std(group_LTurnPC)},
                                                         'boutAmps'           :     {'Mean'    :   np.mean(group_AvgAmps),
                                                                                     'SEM'     :   np.std(group_AvgAmps)},
                                                         'boutDists'          :     {'Mean'    :   np.mean(group_AvgDists),
                                                                                     'SEM'     :   np.std(group_AvgDists)},
                                                         'boutOrts'           :     {'Mean'    :   np.mean(group_AvgOrts),
                                                                                     'SEM'     :   np.std(group_AvgOrts)},
                                                         'cumDist'            :     {'Mean'    :   avgCumDist,
                                                                                     'SEM'     :   SDCumDist},
                                                         'avgHeatmap'         :     {'Mean'    :   avgHeatmap,
                                                                                     'SEM'     :   SDHeatmap},
                                                         'avgBout'            :     {'Mean'    :   groupAvgBout,
                                                                                     'SEM'     :   groupSEMBout},
                                                         'seqProbs1'          :     {'comb'    :   comb1,
                                                                                     'Mean'    :   avg_seqProbS1,
                                                                                     'SEM'     :   se_seqProbS1},
                                                         'seqProbs2_Z'        :     {'comb'    :   comb2,
                                                                                     'Mean'    :   avg_seqProbS2_Z,
                                                                                     'SEM'     :   se_seqProbS2_Z},
                                                        },
                         'ROIs'                      :   seg_data,
                         }
                                                         
    np.save(dictPath_out,GroupedFish) 
    print('Group Dictionary saved at ' + dictPath_out)
    
    
    if createGroupFigures:
        print('Generating group figures, saving at ' + outPathF)
        AZF.groupFigs(GroupedFish,groupName,outPathF)
        outPathROIF=outPathF+'ROIs\\'
        print('Generating group ROI figures, saving at ' + outPathROIF)
        AZU.cycleMkDir(outPathROIF)
        AZF.ROIGroupFigs(GroupedFish,ROI_BPS,ROI_avgVelocity,ROI_avgAngVelocityBout,ROI_biasLeftBout,ROI_LTurnPC,ROI_AvgBoutAmps,ROI_AvgBoutDists,ROI_PCTimeSpent,ROIMasks,groupName,ROINames,outPathROIF)
            