# -*- coding: utf-8 -*-
"""
Created on Mon Nov 04 13:58:42 2019

@author: Tom Ryan (Dreosti Lab, UCL)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Arena Zebrafish Repo
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------
# Set Library Paths
import sys
sys.path.append(lib_path)
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\GitHub\Arena_Zebrafish\ARK\libs'
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import datetime
import glob
import matplotlib.pyplot as plt
import datetime
import scipy.stats as stats
# Import local modules
import AZ_utilities as AZU
import AZ_compare as AZC
import AZ_streakProb as AZP

FPS = 120
folderListFile_Ctrl='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Sham.txt'
folderListFile_Cond='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/Lesion.txt'
FigureFolder='S:/WIBR_Dreosti_Lab/Tom/Data/JuvenileFreeSwimming/B0/NewFigures'#
AZU.cycleMkDir(FigureFolder)
labels=['Control','Lesion']
compName=labels[0]+'_vs_'+labels[1]
dateSuff=(datetime.date.today()).strftime("%y%m%d")
sf=0*60*FPS
ef=-1

# # Set Flags
# overwrite=True
# createDict=True
# createFigures=True
# keepFigures=False
# group=False # if set to TRUE change groupName above (ln 38)
# createGroupFigures=True
# keepGroupFigures=False
# omitForward=False

def getFiles(folderNames):
    trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles=[],[],[],[],[],[]
    for folder in folderNames:
        trackingFolder=folder + r'\\Tracking\\'
        # Grab tracking files from folder or .txt folder list file
        trackingFilest=glob.glob(trackingFolder+'*tracking.npz')
        tailXFilest=glob.glob(trackingFolder+'*SegX.csv')
        tailYFilest=glob.glob(trackingFolder+'*SegY.csv')
        boutFilest=glob.glob(trackingFolder+'Analysis\*bouts*')
        tailFilest=glob.glob(trackingFolder+'Analysis\*tailAnalysis*')
        bodyThetaFilest=glob.glob(trackingFolder+'Analysis\*bodyTheta*')
        for i,s in enumerate(trackingFilest): #
            trackingFiles.append(s)
            tailXFiles.append(tailXFilest[i])
            tailYFiles.append(tailYFilest[i])
            boutFiles.append(boutFilest[i])
            tailFiles.append(tailFilest[i])
            bodyThetaFiles.append(bodyThetaFilest[i])
    return trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles

_,folderNames = AZU.read_folder_list(folderListFile_Ctrl)
_,folderNames1 = AZU.read_folder_list(folderListFile_Cond)
trackingFiles,tailXFiles,tailYFiles,tailFiles,boutFiles,bodyThetaFiles=getFiles(folderNames)
trackingFiles1,tailXFiles1,tailYFiles1,tailFiles1,boutFiles1,bodyThetaFiles1=getFiles(folderNames1)

# Excise bouts tail angle
# Bout PCA
# groupsTheta = [bodyThetaFiles,bodyThetaFiles1]
# groupsBouts = [boutFiles,boutFiles1]
# prewindow=100 # miliseconds around the peak or start to take as 'bout'
# postwindow=400 # miliseconds around the peak or start to take as 'bout'
# for i,groupTheta in enuemrate(groupsTheta):
#     groupBouts=groupsBouts[i]
#     for k,thisBodyTheta in enumerate(group):
#         thisBoutsStarts=groupBouts[k][:,1]
#         for thisBoutStart in thisBoutsStarts:

# cumulative distance
cumDistS,distPerSecS=[],[]
cumDistS1,distPerSecS1=[],[]
for thisTrackingFile in trackingFiles:
    thisTracking=np.load(thisTrackingFile)['tracking']
    fx=thisTracking[:,0]
    fy=thisTracking[:,1]
    [fx_mm],[fy_mm]=AZU.convertToMm([fx], [fy])
    distPerFrame,cumDist=AZU.computeDistPerFrame(fx_mm,fy_mm)
    distPerSecS.append(distPerFrame/FPS)
    cumDistS.append(cumDist)

for thisTrackingFile in trackingFiles1:
    thisTracking=np.load(thisTrackingFile)['tracking']
    fx=thisTracking[:,0]
    fy=thisTracking[:,1]
    [fx_mm],[fy_mm]=AZU.convertToMm([fx], [fy])
    distPerFrame,cumDist=AZU.computeDistPerFrame(fx_mm,fy_mm)
    distPerSecS1.append(distPerFrame/FPS)
    cumDistS1.append(cumDist)

# All first
figname='Cumulative distance - ' + labels[0]
plt.figure(figname)
maxLen=len(cumDistS[0])
minLen=15*60*FPS
sumS,sumS1,keep=[],[],[]
for trace in cumDistS:
    if len(trace)<=maxLen and len(trace)>minLen:
        maxLen=len(trace)
        keep.append(True)
    elif len(trace)>maxLen and len(trace)>minLen:
        keep.append(True)
    elif len(trace)<minLen:
        keep.append(False)
keep=np.array(keep)
numKept=0
for i,trace in enumerate(cumDistS):
    if keep[i]:
        plt.plot(trace[0:maxLen]*0.009,color='b',alpha=0.2)
        numKept+=1
        sumS.append(trace[0:maxLen]*0.009)
summ=np.sum(sumS,axis=0)
cumDist=np.divide(summ,numKept)
plt.plot(cumDist,color='black',linewidth=4)
plt.ylabel('Distance (cm)')
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.savefig(savePath,dpi=600)

figname='Cumulative distance - ' + labels[1]
plt.figure(figname)
maxLen=len(cumDistS1[0])
minLen=15*60*FPS
keep=[]
for trace in cumDistS1:
    if len(trace)<=maxLen and len(trace)>minLen:
        maxLen=len(trace)
        keep.append(True)
    elif len(trace)>maxLen and len(trace)>minLen:
        keep.append(True)
    elif len(trace)<minLen:
        keep.append(False)
keep=np.array(keep)
numKept=0
for i,trace in enumerate(cumDistS1):
    if keep[i]:
        plt.plot(trace[0:maxLen]*0.009,color='b',alpha=0.2)
        numKept+=1
        sumS1.append(trace[0:maxLen]*0.009)
summ1=np.sum(sumS1,axis=0)
cumDist1=np.divide(summ1,numKept)
plt.plot(cumDist1,color='black',linewidth=4)
plt.ylabel('Distance (cm)')
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.savefig(savePath,dpi=600)

# Repeat but conditions together
figname='Cumulative distance'
plt.figure(figname)
for trace in sumS:
    plt.plot(trace,color='b',alpha=0.2,linewidth=2)
plt.plot(cumDist,color='b',linewidth=3,alpha=0.9,label=labels[0])
for trace in sumS1:
    plt.plot(trace,color='r',alpha=0.2,linewidth=2)
plt.plot(cumDist1,color='r',linewidth=3,alpha=0.9,label=labels[1])
plt.xlabel('Time (s)')
plt.xticks(ticks=[0,5*120*60,10*120*60,15*120*60,20*120*60],labels=[0,5,10,15,20])
plt.ylabel('Distance (cm)')
plt.legend()
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.savefig(savePath,dpi=600)

# cumulative angle?
# streaks
angle,seq=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    anglet=thisCtrlBout[:,4]
    # _,seqt=AZP.angleToSeq_1(anglet,keepForward=True)
    # print('Mean seq:'+str(np.mean(seqt)))
    # print('Mean abs seq: '+str(np.mean(np.abs(seqt))))
    for i in anglet: angle.append(i)
    _,seq=AZP.angleToSeq_1(angle,keepForward=True)
    streaksTrue,streaksRandom=AZP.seqToProb(seq)
    
angle,seq=[],[]
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    anglet=thisCondBout[:,4]
    # _,seqt=AZP.angleToSeq_1(anglet,keepForward=True)
    # print('Mean seq:'+str(np.mean(seqt)))
    # print('Mean abs seq: '+str(np.mean(np.abs(seqt))))
    for i in anglet: angle.append(i)
    _,seq=AZP.angleToSeq_1(angle,keepForward=True)
    streaksTrueCond,streaksRandomCond=AZP.seqToProb(seq)

# Report results
print("Controls: {0} vs {1}".format(np.mean(streaksTrue), np.mean(streaksRandom)))
print("Lesions: {0} vs {1}".format(np.mean(streaksTrueCond), np.mean(streaksRandomCond)))

all_streaks_true_controls = np.array(streaksTrue)
all_streaks_random_controls = np.array(streaksRandom)
all_streaks_true_lesions = np.array(streaksTrueCond)
all_streaks_random_lesions = np.array(streaksRandomCond)

hist_streaks_true_controls = np.cumsum(np.histogram(all_streaks_true_controls, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_random_controls = np.cumsum(np.histogram(all_streaks_random_controls, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_true_lesions = np.cumsum(np.histogram(all_streaks_true_lesions, bins = np.arange(-0.5, 16, 1), density=1)[0])
hist_streaks_random_lesions = np.cumsum(np.histogram(all_streaks_random_lesions, bins = np.arange(-0.5, 16, 1), density=1)[0])

tCtrl,pvalueCtrl=stats.ttest_ind(all_streaks_true_controls, all_streaks_random_controls, axis=0, equal_var=False)
tLesion,pvalueLesion=stats.ttest_ind(all_streaks_true_lesions, all_streaks_random_lesions, axis=0, equal_var=False)

chiSq_ctrl=stats.chisquare(f_obs=all_streaks_true_controls, f_exp=all_streaks_random_controls)
chiSq_les=stats.chisquare(f_obs=all_streaks_true_lesions, f_exp=all_streaks_random_lesions)

figname='Streak probability ' + labels[1]
figname='Streak probability'
ylabel='Cumulative Probability'
xlabel='Streak length'

plt.figure(figname)
plt.plot(hist_streaks_true_controls, 'b',label=labels[0])
plt.plot(hist_streaks_random_controls, 'b--',label=labels[0]+' shuffled')
plt.plot(hist_streaks_true_lesions, 'r',label=labels[1])
plt.plot(hist_streaks_random_lesions, 'r--',label=labels[1]+' shuffled')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlim(0,10)
plt.ylim(0,1)
plt.legend()
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.savefig(savePath,dpi=600)

# Proportion of turns

# PCA projection of bouts (tSNE)

# Angle vs Displacement scatter
angle,disp=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    anglet=thisCtrlBout[:,4]
    dispt=thisCtrlBout[:,5]*0.09
    for i in anglet: angle.append(i)
    for i in dispt: disp.append(i)
figname='Angle_vs_Displacement_Sham'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Displacement (mm)'
xlabel='Angle (degrees)'
plt.figure(figname)
plt.title(figname)
plt.scatter(angle,disp,s=2,alpha=0.2)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlim(-100,100)
plt.ylim(0,20)
plt.savefig(savePath,dpi=600)
angle,disp=[],[]
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    anglet=thisCondBout[:,4]
    dispt=thisCondBout[:,5]*0.09
    for i in anglet: angle.append(i)
    for i in dispt: disp.append(i)
figname='Angle_vs_Displacement'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.scatter(angle,disp,s=2,alpha=0.2)
plt.savefig(savePath,dpi=600)
figname='Angle_vs_Displacement_Lesion'
plt.figure(figname)
plt.scatter(angle,disp,s=2,alpha=0.2)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xlim(-100,100)
plt.ylim(0,20)
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
plt.savefig(savePath,dpi=600)

# BPS
save=True
cond,ctrl=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    ctrl.append(np.divide(len(thisCtrlBout),thisCtrlBout[-1,2]/FPS))
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    cond.append(np.divide(len(thisCondBout),thisCondBout[-1,2]/FPS))

figname='BPS'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Bouts per second'
xlabel='GroupName'
yint=[0.5,1.0,1.5,2]
ylim=[0,3]
AZC.compPlot(ctrl,cond,labels,figname,savePath,yint,ylabel,ylim,save=save)

# Durations
cond,ctrl=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    ctrl.append((np.mean(thisCtrlBout[:,3])/FPS)*1000)
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    cond.append((np.mean(thisCondBout[:,3])/FPS)*1000)
figname='Durations'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Bout Duration (ms)'
xlabel='GroupName'
yint=[0,50,100,150,200,250,300,350,400]
ylim=[0,400]
AZC.compPlot(ctrl,cond,labels,figname,savePath,yint,ylabel,ylim,save=save)

# Angle
cond,ctrl=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    ctrl.append(np.mean(np.abs(thisCtrlBout[:,4])))
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    cond.append(np.mean(np.abs(thisCondBout[:,4])))
figname='Angles'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Bout angle (degrees)'
xlabel='GroupName'
yint=[0,10,20,30,40,50,60]
ylim=[0,60]
AZC.compPlot(ctrl,cond,labels,figname,savePath,yint,ylabel,ylim,save=save)

# Displacement
cond,ctrl=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    ctrl.append(np.mean(thisCtrlBout[:,5]*0.09))
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    cond.append(np.mean(thisCondBout[:,5]*0.09))
figname='Displacement'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Bout Displacement (mm)'
xlabel='GroupName'
yint=[0,5,10]
ylim=[0,10]
AZC.compPlot(ctrl,cond,labels,figname,savePath,yint,ylabel,ylim,save=save)

# mean IBI and std
cond,ctrl=[],[]
condvar,ctrlvar=[],[]
for thisCtrlBoutFile in boutFiles:
    thisCtrlBout=np.load(thisCtrlBoutFile)
    ctrlIBI=[]
    for i in range(0,len(thisCtrlBout)-1):
        ctrlIBI.append(thisCtrlBout[i+1,0]-thisCtrlBout[i,1])
    ctrl.append(np.mean(ctrlIBI)/FPS)
    ctrlvar.append(np.std(ctrlIBI)/FPS)
for thisCondBoutFile in boutFiles1:
    thisCondBout=np.load(thisCondBoutFile)
    condIBI=[]
    for i in range(0,len(thisCondBout)-1):
        condIBI.append(thisCondBout[i+1,0]-thisCondBout[i,1])
    cond.append(np.mean(condIBI)/FPS)
    condvar.append(np.std(condIBI)/FPS)
figname='IBI'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Interbout interval (s)'
xlabel='GroupName'
yint=[0,1,2,3,4]
ylim=[0,4.5]
AZC.compPlot(ctrl,cond,labels,figname,savePath,yint,ylabel,ylim,save=save)
figname='IBI Variance'
savePath=FigureFolder+'\\'+figname+'_'+compName+'_'+dateSuff+'.png'
ylabel='Interbout interval variance (s)'
xlabel='GroupName'
yint=[0,5,10,15,20,25]
ylim=[0,25]
AZC.compPlot(ctrlvar,condvar,labels,figname,savePath,yint,ylabel,ylim,save=save)
  

# proportion of turns vs swims

# Excise Bouts







# # # check through to see if analysis has already been done
# # print('Checking existing analysis files...')
# # dictOutNames=[]
# # for i,trackingFile in enumerate(trackingFiles):
# #     d,spl=trackingFile.rsplit(sep='\\',maxsplit=1)
# #     spl=spl[0:-13]
# #     dictFile=DictionaryFolder+spl+'_ANALYSIS.npy'
# #     if os.path.exists(dictFile):
# #         if overwrite==False:
# #             print(trackingFile + ' has already been analysed... removing from list')
# #             trackingFiles.remove(trackingFile)
# #             missingFiles.append(trackingFile)
# #         else:
# #             print(trackingFile + ' has already been analysed... overwriting...')
# #             dictOutNames.append(dictFile)
# #     else:
# #         dictOutNames.append(dictFile)
# # # run through each experiment
# for k,trackingFile in enumerate(trackingFiles):
#     print('Analysing ' + trackingFile)
#     wDir,name,date,gType,cond,chamber,fishNo=AZU.grabFishInfoFromFile(trackingFile)
#     fx,fy,bx,by,ex,ey,area,ort,_=AZU.grabTrackingFromFile(trackingFile)
#     if(ef>len(fx)) : ef = -1
#     fx=fx[sf:ef]
#     fy=fy[sf:ef]
#     bx=bx[sf:ef]
#     by=by[sf:ef]
#     ex=ex[sf:ef]
#     ey=ey[sf:ef]
#     area=area[sf:ef]
#     ort=ort[sf:ef]
#     # How long is this movie?
#     numFrames=fx.shape[0]
#     numSecs=(numFrames/FPS)
#     xRange=np.arange(0,numSecs,(1/FPS))
    
#     # # round down the Fx and Fy coordinates
#     # floorFx=np.floor(fx)
#     # floorFy=np.floor(fy)
    
#     # # make them ints
#     # floorFx=floorFx.astype(int)
#     # floorFy=floorFy.astype(int)
    
#     # # make heatmaps of each fish: a 2D histogram
#     # heatmap, xedges, yedges = np.histogram2d(floorFx, floorFy, bins=10)
    
#     # convert pixels to mm for fx,fy,bx,by,ex and ey
#     [fx_mm,bx_mm,ex_mm],[fy_mm,by_mm,ey_mm] = AZU.convertToMm([fx,bx,ex],[fy,by,ey]) 
    
#     # Compute distance travelled between each frame 
#     distPerFrame,cumDist=AZU.computeDistPerFrame(bx_mm,by_mm)

#     # Check length, looking for shortest in the list over the minimum
#     AZU.checkTracking(distPerFrame)
#     avgVelocity=cumDist[-1] /(len(cumDist)/FPS)   # per second over whole movie
    
#     # load bouts and measure BPS
    
    
#     avgBout = np.mean(allBoutsDist,0)
#     avgBoutSD = np.std(allBoutsDist,0)/np.sqrt(len(allBoutsDist))
#     biasLeftBout = (np.sum(boutAngles))/(np.sum(np.abs(boutAngles))) # positive is bias for left, negative bias for right
#     avgAngVelocityBout = np.mean(np.abs(boutAngles))
#     # Compute bout amplitudes from all bouts peak
#     boutAmps=AZA.findBoutMax(allBoutsDist)
# #    OR
#     # Compute boutAmps from integral of distance travelled during that bout
# #    boutDists=AZA.findBoutArea(allBoutsDist)
    
#     boutDists,_=AZU.computeDistPerBout(bx_mm,by_mm,boutStarts,boutEnds)
#     if omitForward:
#         boutKeep,boutSeq=AZP.angleToSeq_LR(boutAngles)
#         comb1,seqProbs1=AZP.probSeq1(boutSeq,pot=['L','R']) # compute overall observed probability of each bout type (LR)
#         comb2,randProbs2, seqProbs2_V, seqProbs2_Z, seqProbs2_P = AZP.probSeq2(boutSeq,pot=['L','R']) # compute the observed probability of each bout type (FLR) appearing in pairs. Returns labels, raw probabilities and normalised probabilities (relative probability from expected)
#     else:
#         boutSeq=AZP.angleToSeq(boutAngles)
#         comb1,seqProbs1=AZP.probSeq1(boutSeq,pot=['F','L','R']) # compute overall observed probability of each bout type (FLR)
#         comb2,randProbs2, seqProbs2_V, seqProbs2_Z, seqProbs2_P = AZP.probSeq2(boutSeq,pot=['F','L','R']) # compute the observed probability of each bout type (FLR) appearing in pairs. Returns labels, raw probabilities and normalised probabilities (relative probability from expected)
    
#     # Compute the cumulative angle over the movie
#     # avgAngVelocity,bias,cumOrt=AZA.computeCumulativeAngle(ort,plot=False)
# #    AZA.boutHeadings(ort, boutStarts, boutStops)
    
#     params=[]
    
# #    name=name+pstr
#     params.append(  date                )
#     params.append(  gType               )
#     params.append(  cond                )
#     params.append(  chamber             )
#     params.append(  fishNo              )
#     params.append(  trackingFile        )
#     params.append(  AZU.grabAviFileFromTrackingFile(trackingFile))
#     params.append(  BPS                 )
#     params.append(  avgVelocity         )
#     params.append(  distPerFrame        )
#     params.append(  cumDist             )
#     params.append(  avgBout             )
#     params.append(  avgBoutSD           )
#     params.append(  boutAmps            )
#     params.append(  boutDists           )
#     params.append(  boutAngles          )
#     params.append(  heatmap             )
#     params.append(  avgAngVelocityBout  )
#     params.append(  biasLeftBout        )
#     params.append(  LturnPC             )
#     params.append(  boutSeq             )
#     params.append(  seqProbs1           )
#     params.append(  seqProbs2_V         )
#     params.append(  seqProbs2_P         )
#     params.append(  seqProbs2_Z         )
#     params.append(  boutStarts          )
    
#     thisFishDict=AZS.populateSingleDictionary(params=params,allBoutsList=allBouts,allBoutsOrtList=allBoutsOrt,allBoutsDistList=allBoutsDist,comb1=comb1,comb2=comb2)
    
    
#     thisFishDictName=DictionaryFolder+name+'_ANALYSIS' + '.npy'
#     if(createDict):np.save(dictOutNames[k],thisFishDict) 
#     dictList.append(thisFishDict)
#     print('Analysis saved at ' + thisFishDictName)
    
#     if(createFigures):
#         print('Saving figures at ' + figureFolder)
#         AZU.tryMkDir(figureFolder)
#         AZU.tryMkDir(figureFolder+'avgBout\\')
#         AZU.tryMkDir(figureFolder+'HeatMaps\\')
#         AZU.tryMkDir(figureFolder+'CumDist\\')
#         AZU.tryMkDir(figureFolder+'boutAmps\\')
        
#         shtarget=AZF.indAvgBoutFig(avgBout,avgBoutSD,name,figureFolder+'avgBout\\')
        
# #        if(shtarget!=-1):
# #            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
# #        else:
# #            print('WARNING!! Saving figure' + name + '_avgBout failed')
#         ##    
#         shtarget=AZF.indHeatMapFig(heatmap,name,figureFolder+'HeatMaps\\')
        
# #        if(shtarget!=-1):
# #            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
# #        else:
# #            print('WARNING!! Saving figure' + name + '_heatmap failed')
#         ##
#         if(np.sum(cumDist)!=0):
#             shtarget=AZF.indCumDistFig(cumDist,name,figureFolder+'CumDist\\')
#         else:
#             shtarget=-1
            
# #        if(shtarget!=-1):
# #            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
# #        else:
# #            print('WARNING!! Saving figure' + name + '_cumDist failed')
#         if(np.sum(boutAmps)!=0):
#            shtarget=AZF.indBoutAmpsHistFig(boutAmps,name,figureFolder+'boutAmps\\')
#         else:
#             shtarget=-1
# #        if(shtarget!=-1):
# #            AZU.createShortcutTele(shtarget,root=r"D:\\Movies\\Processed\\")
# #        else:
# #            print('WARNING!! Saving figure' + name + '_cumDist failed')
        
#         if(keepFigures==False):plt.close('all')
        
#     ##### END OF FILE LOOP #######
# #FIN
