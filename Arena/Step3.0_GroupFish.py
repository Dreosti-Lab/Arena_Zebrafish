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
import os
import glob

import AZ_figures as AZF
import AZ_utilities as AZU

folderListFile=[]
trackingFolder=[]
trackingSHFolder=[]

# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile = r'D:\Movies\FolderLists\200319.txt'

# OR

#trackingFolder = r'D:\\TrackingData\\Groups\\EC_B2' # for direct link to tracking folder

# OR 

trackingSHFolder = r'D:\\Movies\\RawMovies\\aspB2\\allTrackingToGroup\\' # for a folder of shortcuts

# Set Flags
createGroupFigures=True
keepGroupFigures=True
report=True
omitForward=False
#anSuff='EA_M0_old' #suffix attached to analysis folder (if any) (suff in Step2)
#inSuff='EA_M0_old' # suffix attached to individual dictionaries (if any) (outSuff in Step2.1)
anSuff='' 
inSuff='' 
outSuff='' # suffix to be attached to group dictionary (if any)
groupName='EA_B2'
# Make Dictionary and Figure Folders and define paths
root='D:\\Movies\\RawMovies\\aspB2\\'
inPathD=root+'IndDictionaries\\'
outPathD=root
outPathF=root+'GroupFigures\\'

trackingFiles=[]
dictNameList=[]
dictList=[]
#    groupName=nn + pstr
if(len(folderListFile)!=0 and len(trackingFolder)==0 and len(trackingSHFolder)==0): # then we are dealing with a folderList rather than a folder of shortcuts
    
    # Read Folder List
    ROI_path,folderNames = AZU.read_folder_list(folderListFile)

    # Bulk analysis of all folders
    for idx,folder in enumerate(folderNames):
        AnalysisFolder,_ = AZU.get_analysis_folders(folder)
        
        dicSubFiles = glob.glob(AnalysisFolder + r'\*.npy')
        
        # add to overall list (one by one)
        for s in dicSubFiles:dictNameList.append(s)
        
        
else:
    if(len(folderListFile)==0 and len(trackingSHFolder)!=0 and len(trackingFolder)==0): # then we are dealing with a folder of shortcuts
        
        # cycle through the shortcuts and compile a list of targets
        shFiles=glob.glob(trackingSHFolder+'\*.lnk')
        for i in range(len(shFiles)):
            ret,path=AZU.findShortcutTarget(shFiles[i])
            if(ret==0):
                _,_,f=path.rsplit(sep='\\',maxsplit=2)
                f=f[0:-13]
                dictNameList.append(glob.glob(inPathD + '*' + f + '*.npy')[0])
            else:
                print('Could not find associated dictionary for ' + f)
    else:
        if(len(folderListFile)==0 and len(trackingSHFolder)==0 and len(trackingFolder)!=0): # then we are dealing with a direct tracking folder
            trackingFiles=glob.glob(trackingFolder+'\*.npz')
            for tFile in trackingFiles:
                d,spl=tFile.rsplit(sep='\\')
                spl=spl[0:-13]
                dicChk=inPathD+'\\'+spl+'_ANALYSIS.npy'
                if os.path.exists(dicChk):
                    dictNameList.append(dicChk)
                else:
                    print('Error, missing dictionary for file ' + tFile + '. Removing from the list...')
                    trackingFiles.remove(tFile)
        else:
            sys.exit('No tracking folder, shortcut folder or FolderlistFile provided')
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

for i,f in enumerate(dictNameList):
    dic=np.load(f,allow_pickle=True).item()
    dictList.append(dic)
    data=dic['data']
    if i==0:
        comb1=dic['data']['seqProbs1']['comb']
        comb2=dic['data']['seqProbs2']['comb']
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
        
    mL=(60*60*120)
    lL=(4*60*120)
    shortest=mL
    for cu in group_cumDist:
       if((len(cu)<shortest) and (len(cu)>lL)):shortest=len(cu)
           
    group_cumDistcrop=[]
    for cu in group_cumDist:
        if(len(cu)>= shortest):group_cumDistcrop.append(cu[0:shortest]) # if this is not as long as the minimum then remove (only for the avg group figure purpose)
        
    ###################################################################################################
           
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
                     }
                                                     
np.save(dictPath_out,GroupedFish) 
print('Group Dictionary saved at ' + dictPath_out)


if createGroupFigures:
    print('Generating group figures, saving at ' + outPathF)
    AZF.groupFigs(GroupedFish,groupName,outPathF)
