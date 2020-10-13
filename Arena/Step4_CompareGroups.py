# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:04:43 2020

@author: thoma
"""

# Set Library Paths
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Local\Arena_Zebrafish\libs'

import sys
sys.path.append(lib_path)
import numpy as np
import matplotlib.pyplot as plt
import AZ_utilities as AZU
import AZ_analysis_testing as AZA
import scipy.stats as stats
#import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr
#Rstats = importr('stats')


def compareGroupStats3Dics(dic1File,dic2File,dic3File,FPS=120,save=True,keep=False):
        
    dic1Name,avgCumDistAV_1,avgCumDistSEM_1,avgBoutAmps_1,allBPS_1,avgBoutAV_1,avgBoutSEM_1,avgHeatmap_1,avgVelocity_1=AZA.unpackGroupDictFile(dic1File)
    dic2Name,avgCumDistAV_2,avgCumDistSEM_2,avgBoutAmps_2,allBPS_2,avgBoutAV_2,avgBoutSEM_2,avgHeatmap_2,avgVelocity_2=AZA.unpackGroupDictFile(dic2File)
    dic3Name,avgCumDistAV_3,avgCumDistSEM_3,avgBoutAmps_3,allBPS_3,avgBoutAV_3,avgBoutSEM_3,avgHeatmap_3,avgVelocity_3=AZA.unpackGroupDictFile(dic3File)
#    sF=0
#    eF=36000
#    pstr='0-5min'
#    
#    avgCumDistAV_1
#    avgCumDistSEM_1
#    avgCumDistAV_2
#    avgCumDistSEM_2
#    avgBoutAmps_1
#    avgBoutAmps_2
#    allBPS_1
    ################### avgBout
    saveName=-1
    compName=dic1Name + ' vs ' + dic2Name + 'vs' + dic3Name
    figname='avgBout Comparison. Groups:'+ compName
    plt.figure(figname)
    
    xFr=range(len(avgBoutAV_1))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_1)
    pos1=avgBoutAV_1+avgBoutSEM_1
    neg1=avgBoutAV_1-avgBoutSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgBoutAV_2))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_2)
    pos2=avgBoutAV_2+avgBoutSEM_2
    neg2=avgBoutAV_2-avgBoutSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgBoutAV_3))
    x=np.divide(xFr,FPS)
    plt.plot(x,avgBoutAV_3)
    pos3=avgBoutAV_3+avgBoutSEM_3
    neg3=avgBoutAV_3-avgBoutSEM_3
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### cumDist
    saveName=-1
    xFr=range(len(avgCumDistAV_1))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1)
    pos1=avgCumDistAV_1+avgCumDistSEM_1
    neg1=avgCumDistAV_1-avgCumDistSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2)
    pos2=avgCumDistAV_2+avgCumDistSEM_2
    neg2=avgCumDistAV_2-avgCumDistSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3)
    pos3=avgCumDistAV_3+avgCumDistSEM_3
    neg3=avgCumDistAV_3-avgCumDistSEM_3
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    # Chi-square Test
    shortest=np.min([len(avgCumDistAV_1),len(avgCumDistAV_2)])
    chisq,pvalue=stats.chisquare(avgCumDistAV_1[0:shortest], f_exp=avgCumDistAV_2[0:shortest])
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDist.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ################## cumDist Zoom 15min
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:(63600*2)]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:(63600*2)]
    avgCumDistAV_3Zoom=avgCumDistAV_3[0:(63600*2)]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:(63600*2)]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:(63600*2)]
    avgCumDistSEM_3Zoom=avgCumDistSEM_3[0:(63600*2)]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 15 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3Zoom)
    pos3=avgCumDistAV_3Zoom+avgCumDistSEM_3Zoom
    neg3=avgCumDistAV_3Zoom-avgCumDistSEM_3Zoom
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom15min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### boutAmps
    dBA = [avgBoutAmps_1, avgBoutAmps_2,avgBoutAmps_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgBoutAmps Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Velocity (mm/frame)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgBoutAmps_2, avgBoutAmps_3, axis=0, equal_var=False)
    plt.legend(['p 2 vs 3 = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBoutAmps.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()

    ################## cumDist Zoom 5min
    ll=5*60*FPS
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:ll]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:ll]
    avgCumDistAV_3Zoom=avgCumDistAV_3[0:ll]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:ll]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:ll]
    avgCumDistSEM_3Zoom=avgCumDistSEM_3[0:ll]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 5 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_3Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_3Zoom)
    pos3=avgCumDistAV_3Zoom+avgCumDistSEM_3Zoom
    neg3=avgCumDistAV_3Zoom-avgCumDistSEM_3Zoom
    
    plt.plot(x,neg3,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos3,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg3,pos3,alpha=0.2)
    
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
#    # Fisher's Exact Test
#    rpy2.robjects.numpy2ri.activate()
#    m = np.array([avgCumDistAV_1Zoom,avgCumDistAV_2Zoom])
#    res = stats.fisher_test(m)
#    pvalue=res[0][0]
#    plt.legend(['p = ' + str(round(pvalue,3))])
#    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom5min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()    
    
    #################### avgVel
    dBA = [avgVelocity_1, avgVelocity_2,avgVelocity_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgVelocity Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Velocity (mm/sec)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(avgVelocity_2, avgVelocity_3, axis=0, equal_var=False)
    plt.legend(['p 2 vs 3 = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgVelocity.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### heatmapDiff
    
    avgDiffHeatmap=avgHeatmap_1-avgHeatmap_2
    saveName=-1
    figname='Difference between heatmaps. Groups:'+ compName
    plt.figure(figname)
    plt.imshow(avgDiffHeatmap)
    plt.title(figname)
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_diffHeatmap.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ############### BPS
    dBA = [allBPS_1, allBPS_2, allBPS_3]
    labels=[dic1Name,dic2Name,dic3Name]
    saveName=-1
    figname='avgBPS Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2,3],labels)
    plt.ylabel('Bouts per second')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_2, axis=0, equal_var=False)
    plt.legend(['p 1 vs 2 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_3, axis=0, equal_var=False)
    plt.legend(['p 1 vs 3 = ' + str(round(pvalue,3))])
    t,pvalue=stats.ttest_ind(allBPS_2, allBPS_3, axis=0, equal_var=False)
    plt.legend(['p = 2 vs 3 ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBPS.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    
    
    
def compareGroupStats(dic1File,dic2File,FPS=120,save=True,keep=False):
        
    dic1Name,avgCumDistAV_1,avgCumDistSEM_1,avgBoutAmps_1,allBPS_1,avgBoutAV_1,avgBoutSEM_1,avgHeatmap_1,avgVelocity_1=AZA.unpackGroupDictFile(dic1File)
    dic2Name,avgCumDistAV_2,avgCumDistSEM_2,avgBoutAmps_2,allBPS_2,avgBoutAV_2,avgBoutSEM_2,avgHeatmap_2,avgVelocity_2=AZA.unpackGroupDictFile(dic2File)
#    sF=0
#    eF=36000
#    pstr='0-5min'
#    
#    avgCumDistAV_1
#    avgCumDistSEM_1
#    avgCumDistAV_2
#    avgCumDistSEM_2
#    avgBoutAmps_1
#    avgBoutAmps_2
#    allBPS_1
    ################### avgBout
    saveName=-1
    xFr=range(len(avgBoutAV_1))
    x=np.divide(xFr,FPS)
    compName=dic1Name + ' vs ' + dic2Name
    figname='avgBout Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgBoutAV_1)
    pos1=avgBoutAV_1+avgBoutSEM_1
    neg1=avgBoutAV_1-avgBoutSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    plt.plot(x,avgBoutAV_2)
    pos2=avgBoutAV_2+avgBoutSEM_2
    neg2=avgBoutAV_2-avgBoutSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### cumDist
    saveName=-1
    xFr=range(len(avgCumDistAV_1))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1)
    pos1=avgCumDistAV_1+avgCumDistSEM_1
    neg1=avgCumDistAV_1-avgCumDistSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2)
    pos2=avgCumDistAV_2+avgCumDistSEM_2
    neg2=avgCumDistAV_2-avgCumDistSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    
    # Chi-square Test
    shortest=np.min([len(avgCumDistAV_1),len(avgCumDistAV_2)])
    chisq,pvalue=stats.chisquare(avgCumDistAV_1[0:shortest], f_exp=avgCumDistAV_2[0:shortest])
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDist.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ################## cumDist Zoom 15min
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:(63600*2)]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:(63600*2)]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:(63600*2)]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:(63600*2)]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 15 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom15min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### boutAmps
    dBA = [avgBoutAmps_1, avgBoutAmps_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgBoutAmps Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Velocity (mm/frame)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgBoutAmps_1, avgBoutAmps_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBoutAmps.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()

    ################## cumDist Zoom 5min
    ll=5*60*FPS
    avgCumDistAV_1Zoom=avgCumDistAV_1[0:ll]
    avgCumDistAV_2Zoom=avgCumDistAV_2[0:ll]
    avgCumDistSEM_1Zoom=avgCumDistSEM_1[0:ll]
    avgCumDistSEM_2Zoom=avgCumDistSEM_2[0:ll]
    
    xFr=range(len(avgCumDistAV_1Zoom))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison Zoom 5 min. Groups:'+ compName
    
    plt.figure(figname)
    plt.plot(x,avgCumDistAV_1Zoom)
    pos1=avgCumDistAV_1Zoom+avgCumDistSEM_1Zoom
    neg1=avgCumDistAV_1Zoom-avgCumDistSEM_1Zoom
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2)
    
    xFr=range(len(avgCumDistAV_2Zoom))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2Zoom)
    pos2=avgCumDistAV_2Zoom+avgCumDistSEM_2Zoom
    neg2=avgCumDistAV_2Zoom-avgCumDistSEM_2Zoom
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2)
    plt.title(figname)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/frame)')
    
    # Chi-square Test
    chisq,pvalue=stats.chisquare(avgCumDistAV_1Zoom, f_exp=avgCumDistAV_2Zoom)
    plt.legend(['p = ' + str(round(pvalue,3))])
#    # Fisher's Exact Test
#    rpy2.robjects.numpy2ri.activate()
#    m = np.array([avgCumDistAV_1Zoom,avgCumDistAV_2Zoom])
#    res = stats.fisher_test(m)
#    pvalue=res[0][0]
#    plt.legend(['p = ' + str(round(pvalue,3))])
#    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom5min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()    
    
    #################### avgVel
    dBA = [avgVelocity_1, avgVelocity_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgVelocity Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Velocity (mm/sec)')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(avgVelocity_1, avgVelocity_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgVelocity.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### heatmapDiff
    
    avgDiffHeatmap=avgHeatmap_1-avgHeatmap_2
    saveName=-1
    figname='Difference between heatmaps. Groups:'+ compName
    plt.figure(figname)
    plt.imshow(avgDiffHeatmap)
    plt.title(figname)
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_diffHeatmap.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    ############### BPS
    dBA = [allBPS_1, allBPS_2]
    labels=[dic1Name,dic2Name]
    saveName=-1
    figname='avgBPS Comparison. Groups:'+ compName
    plt.figure(figname)
    plt.boxplot(dBA,notch=False,showfliers=True)
    plt.title(figname)
    plt.xticks([1,2],labels)
    plt.ylabel('Bouts per second')
    plt.xlabel('GroupName')
    
    # Welch's t-test
    t,pvalue=stats.ttest_ind(allBPS_1, allBPS_2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))])
    
    if(save):
        savepath=r'D:\\Movies\\GroupedData\\Comparisons'
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBPS.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    