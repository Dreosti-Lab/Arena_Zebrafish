# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:05:07 2020

@author: thoma

"""

# Set "Library Path" - Arena Zebrafish Repo
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
# Import local modules
import AZ_utilities as AZU
import AZ_analysis as AZA
import scipy.stats as stats
import AZ_figures as AZF


def run(g1,g2,l1='Control',l2='Condition',dicFolder=r'D:\\Analysis\\GroupedData\\Dictionaries\\',savepath=r'D:\\Shelf\\Compare\\',save=True,recompute=False):  
#    g1='EmxGFP_B0_200913'
#    g2='WT_M0_200826'
#    l1='Blank'
#    l2='Maze'
#    save=True
    dic1File=dicFolder+r'\\'+g1 +'.npy'
    dic2File=dicFolder+r'\\'+g2+'.npy'
    print('Comparing group metrics...')
    compareGroupStats(dic1File,dic2File,l1,l2,savepath=savepath)
    dic1=np.load(dic1File,allow_pickle=True).item()
    dic2=np.load(dic2File,allow_pickle=True).item()
#    compareGroupBoutAngleDistributions(dic1,dic2,l1,l2)
    GroupStateProps=AZF.run(g1,g2,l1,l2,dicFolder=dicFolder,savepath=savepath,save=save,recompute=recompute)
    return dic1,dic2,GroupStateProps

#def compareGroupBoutAngleDistributions:
## Utilities for comparing and plotting differences between 2 or three groups
#def groupState(dic1,dic2): ##### PLACEHOLDER (OBVIOUSLY)
#    
#    dicFolder=r'D:\\Movies\\GroupedData\\Dictionaries\\'
#    dic1File=dicFolder+g1 +'.npy'
#    dic2File=dicFolder+g2+'.npy'
#    dicList=[]
#    dicList.append(dic1File)
#    dicList.append(dic2File)
#    print('Running group state figures')
#  
def compPlot(input1,input2,labels,figname,savepath,yint,ylabel,ylim,save=False,SE=False):
    
    plt.figure(figname,constrained_layout=True)
    
    for i in input1:
        plt.scatter(1,i,color='black')
    for i in input2:
        plt.scatter(2,i,color='black')
        
#    plt.boxplot(dBA,notch=False,showfliers=True)
    av1=np.mean(input1)
    av2=np.mean(input2)
    
    # Check if using SE, if not use SD
    if SE:
        se1=np.std(input1)/np.sqrt(len(input1))
        se2=np.std(input2)/np.sqrt(len(input2))
    else:
        se1=np.std(input1)
        se2=np.std(input2)
        
    # plot mean
    point1 = [1, av1]
    point2 = [2, av2]
    x_values = [point1[0], point2[0]] 
    y_values = [point1[1], point2[1]] 
    plt.plot(x_values,y_values,color='black',marker='o',linewidth=3,markersize=12)
    
    # error bars
    point1 = [1, av1-se1]
    point2 = [1, av1+se1]
    x_values = [point1[0], point2[0]] 
    y_values = [point1[1], point2[1]] 
    plt.plot(x_values,y_values,color='black')
    
    point1 = [2, av2-se2]
    point2 = [2, av2+se2]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values,color='black')
    plt.xlim(0,3)
    plt.ylim(ylim)
    
#    plt.title(figname)
    plt.xticks([1,2],labels,fontsize=18)
    plt.yticks(yint,yint,fontsize=18)
    plt.ylabel(ylabel,fontsize=22,labelpad=8)
    # Welch's t-test
    t,pvalue=stats.ttest_ind(input1, input2, axis=0, equal_var=False)
    plt.legend(['p = ' + str(round(pvalue,3))],framealpha=0,handlelength=0,fontsize=18)
    
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    
    if save:
        plt.savefig(savepath,dpi=600)
    
def compareGroupStats(dic1File,dic2File,l1,l2,savepath=r'D:\\Analysis\\GroupedData\\',FPS=120,save=True,keep=False,recompute=False,col1='#486AC6',col2='#F3930C'):
    savepath=savepath+r'\\Comparisons'    
    dic1Name,avgCumDistAV_1,avgCumDistSEM_1,avgBoutAmps_1,allBPS_1,avgBoutAV_1,avgBoutSEM_1,avgHeatmap_1,avgVelocity_1,avgAngVelocityBouts_1,biasLeftBout_1,LTurnPC_1        =   AZA.unpackGroupDictFile(dic1File)    
    dic2Name,avgCumDistAV_2,avgCumDistSEM_2,avgBoutAmps_2,allBPS_2,avgBoutAV_2,avgBoutSEM_2,avgHeatmap_2,avgVelocity_2,avgAngVelocityBouts_2,biasLeftBout_2,LTurnPC_2        =   AZA.unpackGroupDictFile(dic2File)    
    compName=dic1Name + ' vs ' + dic2Name
    #################### avgAngVel
    input1=avgAngVelocityBouts_1
    input2=avgAngVelocityBouts_2
    saveName=-1
    figname='Turn Bias Index. Groups:'+ compName
    labels=[l1,l2]
    saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgAngVelocity.png'
    ylabel='Angular Velocity (mm/sec)'
    yint=(0,10,20,30,40)
    ylim=(0,40)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
    if(keep==False):plt.close()
    
    ####################  biasLeftBout
    input1=biasLeftBout_1
    input2=biasLeftBout_2
    saveName=-1
    figname='Left turn bias. Groups:'+ compName
    saveName=savepath+r'\\'+compName + r'\\' + compName + '_LTurnBias.png'
    ylabel='Turn bias Index'
    yint=(-0.2,0,0.2)
    ylim=(-0.25,0.25)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
    if(keep==False):plt.close()
    
    ## LTurnPC
    input1=LTurnPC_1
    input2=LTurnPC_2
    saveName=-1
    figname='Left turn PC. Groups:'+ compName
    saveName=savepath+r'\\'+compName + r'\\' + compName + '_LTurnPC.png'
    ylabel='Left turn %'
    yint=(40,50,60)
    ylim=(35,65)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
    if(keep==False):plt.close()
    
    ################### avgBout
    saveName=-1
    xFr=range(len(avgBoutAV_1))
    x=np.divide(xFr,FPS)
    yint=(0,5,10,15)
    ylim=(0,15)
    xint=(0,0.1,0.2,0.3,0.4)
    xlabel=(0,100,200,300,400)
    ylabel='Velocity (mm/s)'
    figname='avgBout Comparison. Groups:'+ compName
    plt.figure(figname,constrained_layout=True)
    plt.plot(x,avgBoutAV_1,label=labels[0],color=col1)
    pos1=avgBoutAV_1+avgBoutSEM_1
    neg1=avgBoutAV_1-avgBoutSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2,color=col1)
    
    plt.plot(x,avgBoutAV_2,label=labels[1],color=col2)
    pos2=avgBoutAV_2+avgBoutSEM_2
    neg2=avgBoutAV_2-avgBoutSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2,color=col2)
    plt.xlabel('Time (ms)',fontsize=18,labelpad=8)
    plt.ylabel(ylabel,fontsize=18,labelpad=8)
    plt.xticks(xint,xlabel,fontsize=18)
    plt.yticks(yint,yint,fontsize=18)
    plt.legend(fontsize=18,handlelength=1,framealpha=0)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    
    if(save):
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgBout.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### cumDist
    saveName=-1
    xFr=range(len(avgCumDistAV_1))
    x=np.divide(xFr,FPS)
    figname='cumDist Comparison. Groups:'+ compName
    plt.figure(figname,constrained_layout=True)
    plt.plot(x,avgCumDistAV_1,color=col1,label=labels[0])
    pos1=avgCumDistAV_1+avgCumDistSEM_1
    neg1=avgCumDistAV_1-avgCumDistSEM_1
    
    plt.plot(x,neg1,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos1,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg1,pos1,alpha=0.2,color=col1)
    
    xFr=range(len(avgCumDistAV_2))
    x=np.divide(xFr,FPS)
    
    plt.plot(x,avgCumDistAV_2,color=col2,label=labels[1])
    pos2=avgCumDistAV_2+avgCumDistSEM_2
    neg2=avgCumDistAV_2-avgCumDistSEM_2
    
    plt.plot(x,neg2,color='black',linewidth=0.5,alpha=0.7)
    plt.plot(x,pos2,color='black',linewidth=0.5,alpha=0.7)
    plt.fill_between(x,neg2,pos2,alpha=0.2,color=col2)
#    plt.title(figname)
    plt.xlabel('Time (min)',fontsize=18)
    plt.ylabel('Distance (cm)',fontsize=18)
    xint=(0,600,1200,1800,2400,3000,3600)
    xlabels=(0,10,20,30,40,50,60)
    yint=(0,2000,4000,6000,8000,10000,12000)
    ylabels=(0,20,40,60,80,100,120)
    plt.xticks(xint,xlabels,fontsize=18)
    plt.yticks(yint,ylabels,fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2,length=6)
    ax.yaxis.set_tick_params(width=2,length=6)
    plt.legend(fontsize=18,framealpha=0)
    # Chi-square Test
    shortest=np.min([len(avgCumDistAV_1),len(avgCumDistAV_2)])
    chisq,pvalue=stats.chisquare(avgCumDistAV_1[0:shortest], f_exp=avgCumDistAV_2[0:shortest])
#    plt.legend(['p = ' + str(round(pvalue,3))],fontsize=18,handlelength=0,framealpha=0)
    
    if(save):
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
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom15min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    #################### boutAmps
    input1=avgBoutAmps_1
    input2=avgBoutAmps_2
    saveName=-1
    figname='avgBoutAmps Comparison. Groups:'+ compName
    saveName=savepath+r'\\'+compName + r'\\' + compName + 'avgBoutAmps.png'
    ylabel='Bout amplitude (AU)'
    yint=(0,5,10,15,20,25,30)
    ylim=(0,30)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
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
    plt.ylabel('Distance (mm)')
    
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
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_cumDistZoom5min.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()    
    
    #################### avgVel
    input1=avgVelocity_1
    input2=avgVelocity_2
    saveName=-1
    figname='avgVelocity Comparison. Groups:'+ compName
    saveName=savepath+r'\\'+compName + r'\\' + compName + '_avgVelocity.png'
    ylabel='Average Velocity (mm/sec)'
    yint=(0,1,2,3,4,5,6)
    ylim=(0,6)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
    if(keep==False):plt.close()
    
    ############### BPS
    input1=allBPS_1
    input2=allBPS_2
    saveName=-1
    figname='avgBPS Comparison. Groups:'+ compName
    saveName=savepath+r'\\'+compName + r'\\' + compName + '_BPS.png'
    ylabel='Bouts per second'
    yint=(0,0.5,1,1.5)
    ylim=(0,1.5)
    if(save):
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=True)
    else:
        compPlot(input1,input2,labels,figname,saveName,yint,ylabel,ylim,save=False)
    if(keep==False):plt.close()
    
    #################### heatmapDiff
    avgDiffHeatmap=avgHeatmap_1-avgHeatmap_2
    saveName=-1
    figname='Difference between heatmaps. Groups:'+ compName
    plt.figure(figname)
    plt.imshow(avgDiffHeatmap)
    plt.title(figname)
    
    if(save):
        saveName=savepath+r'\\'+compName + r'\\' + compName + '_diffHeatmap.png'
        AZU.cycleMkDir(savepath+r'\\'+compName +r'\\')
        plt.savefig(saveName,dpi=600)
        
    if(keep==False):plt.close()
    
    
    # 3 dictionary function not currently functional 
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
#run(g1,g2,l1=l1,l2=l2,save=True,dicFolder=dicFolder,savepath=savepath)

