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
import glob
import AZ_figures as AZF

FPS=120
folderListFile=[]
trackingFolder=[]

# Specify Folder List of original files OR define the path to the tracking data shortcut folder
#folderListFile = r'D:\Movies\FolderLists\200319.txt'

# OR

trackingFolder = r'D:\\Movies\\GroupedData\\Groups\\AllCtrl\\'

# Set Flags
createGroupFigures=True
keepGroupFigures=True

groupName='AllCtrl'

# Make Dictionary and Figure Folders and define output paths
outPathD='D:\\Movies\\GroupedData\\Dictionaries\\'
outPathF='D:\\Movies\\GroupedData\\Figures\\'

trackingFiles=[]
dictNameList=[]
dictList=[]
#    groupName=nn + pstr
if(len(folderListFile)!=0 and len(trackingFolder)==0): # then we are dealing with a folderList rather than a folder of shortcuts
    
    # Read Folder List
    ROI_path,folderNames = AZU.read_folder_list(folderListFile)

    # Bulk analysis of all folders
    for idx,folder in enumerate(folderNames):
        AnalysisFolder,_ = AZU.get_analysis_folders(folder)
        
        dicSubFiles = glob.glob(AnalysisFolder + r'\*.npy')
        
        # add to overall list (one by one)
        for s in dicSubFiles:dictNameList.append(s)
        
        
else:
    
    if(len(folderListFile)==0 and len(trackingFolder)!=0): # then we are dealing with a folder of shortcuts
        
        # cycle through the shortcuts and compile a list of targets
        shFiles=glob.glob(trackingFolder+'\*.lnk')
        for i in range(len(shFiles)):
            ret,path=AZU.findShortcutTarget(shFiles[i])
            if(ret==0):
                d,_,f=path.rsplit(sep='\\',maxsplit=2)
                AnalysisFolder=d + '\\Analysis\\'
                f=f[0:-13]
                dictNameList.append(glob.glob(AnalysisFolder + r'\\*' + f + '*ANALYSIS.npy'))
            else:
                print('Could not find associated dictionary for ' + f)
    else:
        if(len(folderListFile)==0 and len(trackingFolder)==0):
            sys.exit('No tracking folder or FolderlistFile provided')
            
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
group_boutAmps=[]
group_boutOrts=[]

group_Amps=[]
group_Orts=[]

for f in dictNameList:
    f=f[0]
    dic=np.load(f,allow_pickle=True).item()
    dictList.append(dic)
    data=dic['data']
    
    group_BPS.append(data['BPS'])
    group_avgVelocity.append(data['avgVelocity'])
    group_avgAngVelocityBout.append(data['avgAngVelocityBout'])
    group_biasLeftBout.append(data['biasLeftBout']) 
    group_LTurnPC.append(data['LTurnPC'])
    group_boutAmps.append(np.mean(data['boutAmps']))
    group_boutOrts.append(np.mean(data['boutOrts']))
    group_cumDist.append(data['cumDist'])
    group_heatmap.append(data['heatmap'])
    group_avgBout.append(data['avgBout']['Mean'])
    
    for i,av in enumerate(group_boutAmps):
        group_Amps.append(np.mean(av))
        group_Orts.append(np.mean(group_boutOrts[i]))
        
    mL=(60*60*FPS)
    lL=(45*60*FPS)
    shortest=mL
    for cu in group_cumDist:
       if((len(cu)<shortest) and (len(cu)>lL)):shortest=len(cu)
           
    group_cumDistcrop=[]
    for cu in group_cumDist:
        if(len(cu)>= shortest):group_cumDistcrop.append(cu[0:shortest]) # if this is not as long as the minimum then remove (only for the avg group figure purpose)
           
avgCumDist=(np.mean(group_cumDistcrop,axis=0)) # cumDistAV
SDCumDist=(np.std(group_cumDistcrop, axis=0)) # cumDistSEM
avgHeatmap=(np.mean(group_heatmap,axis=0)) # avgHeatmap
SDHeatmap=(np.std(group_heatmap,axis=0)) # SEMHeatmap
groupAvgBout=(np.mean(group_avgBout, axis=0)) # avgBoutAV
groupSEMBout=(np.std(group_avgBout, axis=0)) # avgBoutSEM
    
# Pool all the metrics for which there is a single value per fish
    
GroupedFish    =    {'Name'                     :   groupName,
                     'Ind_fish'                 :   dictList,
                     'PooledData'               :   {'BPSs'               :   group_BPS,
                                                     'avgVelocity'        :   group_avgVelocity,
                                                     'avgAngVelocityBout' :   group_avgAngVelocityBout,
                                                     'biasLeftBout'       :   group_biasLeftBout,
                                                     'LTurnPC'            :   group_LTurnPC,
                                                     'boutAmps'           :   group_boutAmps,
                                                     'boutOrts'           :   group_boutOrts,
                                                     'cumDists'           :   group_cumDist,
                                                     'avgHeatmaps'        :   group_heatmap,
                                                     'avgBouts'           :   group_avgBout,
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
                                                     'boutAmps'            :    {'Mean'    :   np.mean(group_Amps),
                                                                                 'SEM'     :   np.std(group_Amps)},
                                                     'boutOrts'            :    {'Mean'    :   np.mean(group_Orts),
                                                                                 'SEM'     :   np.std(group_Orts)},
                                                     'cumDist'            :     {'Mean'    :   avgCumDist,
                                                                                 'SEM'     :   SDCumDist},
                                                     'avgHeatmap'         :     {'Mean'    :   avgHeatmap,
                                                                                 'SEM'     :   SDHeatmap},
                                                     'avgBout'            :     {'Mean'    :   groupAvgBout,
                                                                                 'SEM'     :   groupSEMBout},
                                                     },
                     }
                                                     
np.save(dictPath_out,GroupedFish) 
print('Group Dictionary saved at ' + dictPath_out)


if createGroupFigures:
    print('Generating figures, saving at ' + outPathF)
    AZF.groupFigs(GroupedFish,groupName,outPathF)