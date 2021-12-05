# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 09:21:29 2019

@author: Tom Ryan (Dreosti Lab, UCL)
"""
# -----------------------------------------------------------------------------

lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
import math
# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------

# Import useful libraries
import AZ_video_Juvenile as AZVJ
import AZ_math as AZM
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from win32com.client import Dispatch
import glob
import pandas as pd
import scipy.ndimage
import imageio

 # find difference before stimulus
def makeDiffImg(folder,vid,startFrame,numFrames,stepFrames,previous,height,width,bgSuff='',diffSuff=''):
     
    bFrames = 50
    thresholdValue=10
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    print(str(startFrame))
    print(str(numFrames))
    print(str(stepFrames))
     
    for i, f in enumerate(range(int(startFrame), int(numFrames), int(stepFrames))):
   
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
        previous = current
  
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
         
        bCount=0
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
   
        print (numFrames-f)
        print (bCount)

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)

    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)
    
    # Compute Background Frame (median or mode)
    background = np.uint8(np.median(backgroundStack, axis = 2))

    saveFolder = folder
    imageio.imwrite(saveFolder + r'/difference' + diffSuff + '.png', equ)    
    imageio.imwrite(saveFolder + r'/background' + bgSuff + '.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0
     
def pre_process_video_OMR(folder):
    
     FPS=120
     # Load Video
     aviFiles = glob.glob(folder+'/*.avi')
     aviFile = aviFiles[0]
     vid = cv2.VideoCapture(aviFile)
     
     stimFiles = glob.glob(folder+'/*.csv')
     stimFile = stimFiles[0]
     stim =pd.DataFrame.to_numpy(pd.read_csv(stimFile, sep=',',header=None))
     stimStart=stim[0,0]*FPS
     stimDuration=(stim[0,1]*FPS)
     # Ori=np.round(stim[0,4]/(np.pi/180)) # 90 is left on the movie (away from heat)
         
     numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
     
     # Read First Frame
     ret, im = vid.read()
     previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
     width = np.size(previous, 1)
     height = np.size(previous, 0)
   
     # Alloctae Image Space
     stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
            
     # make overall background and difference images
     makeDiffImg(folder,vid,0,numFrames,stepFrames,previous,height,width)
     # make pre-stim background and difference images
     makeDiffImg(folder,vid,int(stimStart),int(stimDuration),stepFrames,previous,height,width,bgSuff='preStim',diffSuff='preStim')
     # make post-stim background and difference images
     makeDiffImg(folder,vid,stimStart+stimDuration,numFrames-(stimStart+stimDuration),stepFrames,previous,height,width,bgSuff='postStim',diffSuff='postStim')
     return 0


def subpixel_intensity(image, x, y):
    xi = np.int(np.round(x))
    yi = np.int(np.round(y))
    dx = x - xi
    dy = y - yi
    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    accum = 0.0
    accum += weight_tl * image[yi, xi]
    accum += weight_tr * image[yi, xi+1]
    accum += weight_bl * image[yi+1, xi]
    accum += weight_br * image[yi+1, xi+1]
    return accum

# Get intensity profile along arc
def arc_profile(image, x, y, radius, start_theta, stop_theta, step_theta):
    profile = []
    xs = []
    ys = []
    for t in np.arange(start_theta, stop_theta, step_theta):
        nx = x + (radius * np.cos(t))
        ny = y + (radius * np.sin(t))
        value = subpixel_intensity(image, nx, ny)
        xs.append(nx)
        ys.append(ny)
        profile.append(value)    
    return np.array(profile), np.array(xs), np.array(ys)



# debug script

def testArc(xo,yo,heading,diffimg,arcLen=90,arcRad=7,arcSam=2.5):
    
    mask=np.copy(diffimg)
    arc=findArc(xo,yo,heading,arcLen=arcLen,arcRad=arcRad,arcSam=arcSam)
    mask[int(xo),int(yo)]=255
    for a in arc:
        mask[int(a[0]),int(a[1])]=128
    [h,w]=diffimg.shape
    arcVals=getArcVals(arc,diffimg,w,h)
    arcFilt=scipy.ndimage.gaussian_filter1d(arcVals, 5)
    peak=arc[np.argmax(arcFilt)]
    mask[int(peak[0]),int(peak[1])]=255
#    plt.imshow(mask)
    return arc,peak,mask

def getArcVals(arc,img,w,h):
    arcVals=[]
    for idx,a in enumerate(arc):
#        print('n='+str(idx))
        if a[0]>=h: a[0]=h-2
        if a[1]>=w: a[1]=w-2
        arcVals.append(getSubPixelIntensity(a,img))
    return arcVals

def findArc(xo,yo,heading,img,arcLen=90,arcRad=7,arcSam=2.5,debugFlag=False):
    conv=np.pi/180
    # heading comes in between -180 and +180
    if heading<0:
        heading+=360
    
    # flip the heading
    if heading<180:
        vec=heading+180
    else:
        vec=heading-180
    
    # define start and end angles for arc, convert to radians
    radInt=arcSam*conv
    startArcAngle=(vec-(arcLen*0.5))
    if startArcAngle<0:
        startArcAngle+=360
    startArc=startArcAngle*conv
    endArcAngle=(vec+(arcLen*0.5))
    if endArcAngle>360:
        endArcAngle-=360
    endArc=endArcAngle*conv
    # debug
    # find number of angle samples
    numSam=int(np.round(np.divide(arcLen,arcSam)))
    # find range of angles to find along arc (swap start and end if end is smaller than start)
#    if np.abs(np.subtract(endArcAngle,startArcAngle))>100:
#        print('heading='+str(heading))
#        print('vec='+str(vec))
#        print('arcLen='+str(arcLen))
#        print('actualLen='+str(np.abs(np.subtract(endArcAngle,startArcAngle))))
#        print('startAngle='+str(startArcAngle))
#        print('endAngle='+str(endArcAngle))
#    if startArcAngle>endArcAngle:
#        interval=arcSam*-1
#    else:
#        interval=arcSam
#    angles = np.arange(startArcAngle, endArcAngle, interval)
#    rads=np.linspace(startArc,endArc,numSam)
    rads=[]
    rads.append(startArcAngle*conv)
    for i in range(1,numSam):
        rT=rads[i-1]+radInt
        if rT>(np.pi*2):
            rT-=(np.pi*2)
        rads.append(rT)
    points_list=[]
    xs=[]
    ys=[]
    for a in rads:
        x=xo + (np.sin(a) * arcRad)
        y=yo + (np.cos(a) * arcRad)
        # reflect x (actually y) along a horizontal line from the origin x (actually y)
        x-=2*np.subtract(x,xo)
        xs.append(x)
        ys.append(y)
        points_list.append([x,y]) 
    [h,w]=img.shape
    arcVals=getArcVals(points_list,img,w,h)
#    points_list=removeDuplicates(points_list)
    return arcVals,np.asarray(points_list)[:,1],np.asarray(points_list)[:,0]
    
# Scripts to find circle edges given origin and radius
def removeDuplicates(lst):
      
    return [t for t in (set(tuple(i) for i in lst))]

def findCircleEdgeSubPix(xo=64,yo=64,r=5,sampleN=144):

    rads = np.linspace(0, 2 * np.pi, sampleN, endpoint=False)
    points_list=[]
    for a in rads:
        x=xo + np.sin(a) * r
        y=yo + np.cos(a) * r
        points_list.append([x,y]) 
        
#    points_list=removeDuplicates(points_list)
    return points_list

def findCircleEdge(xo=64,yo=64,r=5,sampleN=144):

    rads = np.linspace(0, 2 * np.pi, sampleN, endpoint=False)
    points_list=[]
    for a in rads:
        x=int(np.round(xo + np.sin(a) * r))
        y=int(np.round(yo + np.cos(a) * r))
        points_list.append([x,y]) 
        
#    points_list=removeDuplicates(points_list)
    return points_list

def getSubPixelIntensity(a,image): # only works one pixel radius... consider convolutional method for variable kernel sizes
    
    [x,y]=a
    # round coordinates to find root pixel
    xR=np.int(np.round(x))
    yR=np.int(np.round(y))
    dims=image.shape
    # triple check we're not at the edge
    if xR>=dims[1]-1:xR=dims[1]-2
    if yR>=dims[0]-1:yR=dims[0]-2
        
    # mod to 1 (how much closer to the adjacent pixel am I?)
    remX=np.mod(x,1)
    remY=np.mod(y,1)
    
    # find how much closer you are to the diagonal pixel
    remD=np.sqrt(((1-remX)**2)+((1-remY)**2))
    
    # decide which direction we go to find adjacent pixels
    if x-xR<0:
        adjX=xR-1
    else:
        adjX=xR+1
    if y-yR<0:
        adjY=yR+1
    else:
        adjY=yR-1
    
    # find intensity of root pixel
    iR=image[xR,yR]

    # weight root and add weighted points in each direction...
#    check=[adjX,adjY,yR,xR]
#    if np.max(check)>dims[0]-1:
#        print('here')
    valX=(iR*remX)+(image[adjX,yR]*(1-remX))
    valY=(iR*remY)+(image[xR,adjY]*(1-remY))
    valD=(iR*remD)+(image[adjX,adjY]*(1-remD))
    
    # return the mean
    return np.mean([valX,valY,valD])


def getBresenhamOctantPixels(xc,yc,x,y,symPixList=[]):
    symPixList.append([xc+x, yc+y])
    symPixList.append([xc-x, yc+y])
    symPixList.append([xc+x, yc-y])
    symPixList.append([xc-x, yc-y])
    symPixList.append([xc+y, yc+x])
    symPixList.append([xc-y, yc+x])
    symPixList.append([xc+y, yc-x])
    symPixList.append([xc-y, yc-x])

    return symPixList

def discretiseAngleVector(dAngle):
    
    out=[]
    pturn=0
    for i in dAngle:
        if i > 10:
            pturn+=1
            out.append(1)
        elif i < -10:
            pturn+=1
            out.append(-1)
        else: 
            out.append(0)
    pturn= pturn / len(dAngle)
        
    return out,pturn

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v
    
# Extract loom starts and end from stimFile generated by bonsai
def findLoomsFromFile(filePath,responseSeconds=0.5,windowSeconds=3,FPS=120,lim=432000):
    
    loomStarts=[]
    loomEnds=[]
    respEnds=[]
    loomPosX=[]
    loomPosY=[]
    
    responseFrames=np.int(responseSeconds*FPS)
    windowFrames=np.int(windowSeconds*FPS)
    print(filePath)
    df=pd.DataFrame.to_numpy(pd.read_csv(filePath, sep=',',header=None))
    
    i=0
    while i<lim:
        if df[i,1]!=0 and not np.isnan(df[i,1]): # if the loom has been triggered
            if (i + windowFrames)<lim:
                loomStarts.append(i)
                loomEnds.append(i+windowFrames)
                respEnds.append(i+responseFrames)
                loomPosX.append(df[i,2])
                loomPosY.append(df[i,3])
            i=i+windowFrames
        i=i+1
    print('No. of Looms:' + str(len(loomStarts)))
    return loomStarts,loomEnds,respEnds,loomPosX,loomPosY

### Group Dictionary functions ###

def getTrackingFilesFromGroupDict(dic):
    files=[]
    for i in dic['Ind_fish']:
        files.append(i['info']['TrackingPath'])
    return files

def collectIndFishDatafromGroupDict(groupDic,measure=''):
# Collects data of the input measure from all individual fish of a group dictionary
    met=[]
    if measure=='':
        print('Default measure is BPS, use string of desired measure otherwise as collectIndFishDatafromGroupDic(groupDic,measure="")')
        measure='BPS'
    for i in range(0,len(groupDic['Ind_fish'])):
        met.append(groupDic['Ind_fish'][i]['data'][measure])
    return met
    
## File structure and info functions ##########################################

def tryMkDir(path,report=0):
## Creates a new folder at the given path
## returns -1 and prints a warning if fails
## returns 1 if passed
    
    try:
        os.mkdir(path)
    except OSError:
        if(report):
            print ("Creation of the directory %s failed" % path + ", it might already exist!")
        return -1
    else:
        if(report):
            print ("Successfully created the directory %s " % path)
        return 1
    
def cycleMkDir_forw(path,report=0):
## Creates folders and subfolder along defined path
## returns -1 and prints a warning if fails
## returns 1 if passed
    splitPath=path.split(sep="/")
    for i,name in enumerate(splitPath):
        if(i==0):
            s=name+"/"
        else:
            s=s+name+"/"
        if(i!=len(splitPath)-1):    
            tryMkDir(s,report=report)
        else:
            tryMkDir(s,report=report)   
            
def cycleMkDir(path,report=0):
## Creates folders and subfolder along defined path
## returns -1 and prints a warning if fails
## returns 1 if passed
    splitPath=path.split(sep=r"\\")
    for i,name in enumerate(splitPath):
        if(i==0):
            s=name+r"\\"
        else:
            s=s+name+r"\\"
        if(i!=len(splitPath)-1):    
            tryMkDir(s,report=report)
        else:
            tryMkDir(s,report=report)
        
def cycleMkDirr(path,report=0):
## Alternate version in case not 'real' strings used   
    splitPath=path.split(sep="\\")
    for i,name in enumerate(splitPath):
        if(i==0):
            s=name+"\\"
        else:
            s=s+name+"\\"
        if(i<len(splitPath)-1):    
            tryMkDir(s,report=report)
            
def findShortcutTarget(path):    
## finds and returns the target of any shortcut given its path
## INPUTS: path - full path (as a string) of shortcut file
## OUTPUTS: returns -1 if no shortcut is found
##          returns 0,and target path as a string otherwise.
    
    shell = Dispatch("WScript.Shell")
    try:
        shortcut = shell.CreateShortCut(path)
    except OSError:
        return -1,"_"
    else:
        return 0,shortcut.Targetpath

def createShortcut(target,location):
## Creates a shortcut to given target at given location
    locationFolder,shName=location.rsplit(sep="\\",maxsplit=1)
    if(os.path.exists(locationFolder)==False):
            a=tryMkDir(locationFolder,report=0)
            if(a==-1):
                cycleMkDir(locationFolder)
    
    location=locationFolder + "\\" + shName  
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(location)
    shortcut.Targetpath = target
    shortcut.IconLocation = target
    shortcut.save() 
    
def getTrackingFilesFromFolder(suff='',folderListFile=[],trackingFolder=[]):
## Checks an input folder for tracking shortcuts or folderList txt file and collates a list of tracking files to perform analyses on
## Returns an iterable list of file paths for files that exist on the txt file paths or shortcut targets        
    trackingFiles=[]
    
    if(len(folderListFile)!=0 and len(trackingFolder)==0): # then we are dealing with a folderList rather than a folder of shortcuts
        
        # Read Folder List
        ROI_path,folderNames = read_folder_list(folderListFile)
    
        # Build list of files
        for idx,folder in enumerate(folderNames):
            AnalysisFolder,_,TrackingFolder = get_analysis_folders(folder)
            AnalysisFolder=AnalysisFolder + suff
            
            # List tracking npzs
            trackingsubFiles = glob.glob(TrackingFolder + r'\*.npz')
            
            # add to overall list (one by one)
            for s in trackingsubFiles:trackingFiles.append(s)
            
            # Make analysis folder for each data folder
            cycleMkDir(AnalysisFolder)
           
    else:
        
        if(len(folderListFile)==0 and len(trackingFolder)!=0): # then we are dealing with a folder of shortcuts
            
            # cycle through the shortcuts and compile a list of targets
            shFiles=glob.glob(trackingFolder+'\*.lnk')
            for i in range(len(shFiles)):
                ret,path=findShortcutTarget(shFiles[i])
                if(ret==0):
                    trackingFiles.append(path)
                
        else:
            if(len(folderListFile)==0 and len(trackingFolder)==0):
                sys.exit('No tracking folder or FolderlistFile provided')
                
    # -----------------------------------------------------------------------
    numFiles=len(trackingFiles)
    if(numFiles==0):
        print('No Tracking files found... check your path carefully')     
    return trackingFiles
                
def createShortcutTele(target,root=[],location="default",types=["tracking","cropped","initial_background","final_background","initial_tracking","final_tracking","avgBout","heatmap","cumDist","boutAmpsHist"]):
## Automatically builds experiment folder structure to facilitate grouping of experiments based on fileName convention
    if(location=="default"):
        spl=target.split(sep="_")
    
        if(len(spl)!=6 and len(spl)!=7):
            message="File naming system inconsistent for target ## " + target + " ##. Skipping shortcut creation as I don't know where to put it!"
            print(message)
            return
        
        w=spl[0]
        wDir,date=w.rsplit(sep="\\",maxsplit=1)
        spl=spl[1:]
        gType=spl[0]
        cond=spl[1]
        chamber=spl[2]
        trial=spl[3]
        
        if(len(spl)==5):
            typ,ext=spl[4].split(sep=".")
        else:
            e,ext=spl[5].split(sep=".")
            typ=spl[4] + "_" + e
        
        filename=date+r"_"+gType+r"_"+cond+r"_"+chamber+r"_"+trial
        
        locationFolder=wDir + gType + r"_" + cond + r"_" + chamber + r'\\'
                
#        t=trackingSwitcher(typ)
        if(typ==types[0]):
            a=r"Tracking\\"
        elif(typ==types[1]):
            a=r"CroppedMovies\\"
        elif(typ==types[2]):
            a=r"Tracking\\Figures\\InitialBackGround\\"
        elif(typ==types[3]):
            a=r"Tracking\\Figures\\FinalBackGround\\"
        elif(typ==types[4]):
            a=r"Tracking\\Figures\\InitialTracking\\"
        elif(typ==types[5]):
            a=r"Tracking\\Figures\\FinalTracking\\"
        elif(typ==types[6]):
            a=r"Analysis\\Figures\\avgBout\\"
        elif(typ==types[7]):
            a=r"Analysis\\Figures\\HeatMaps\\"
        elif(typ==types[8]):
            a=r"Analysis\\Figures\\cumDist\\"
        elif(typ==types[9]):
            a=r"Analysis\\Figures\\BoutAmps\\"
        else:
            a=typ + r"\\"
            
        l=locationFolder+a
        if(os.path.exists(l)==False):
            cycleMkDir(l,report=0)
        
        location=l+filename+r"_"+typ+'.lnk'
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(location)
    shortcut.Targetpath = target
    shortcut.IconLocation = target
    shortcut.save()

def trackingSwitcher(i,types=["tracking","cropped","initial_background","final_background","initial_tracking","final_tracking"]):
## Used to distinguish and parse different figures in the same folder... can't remember how it's implemented
    for k in range(len(types)):
        switcher={k:types[k]}
    
    return switcher.get(i,-1)

def grabTrackingFromFile(trackingFile,sf=0,ef=-1):
## Loads tracking data from given path 
    data = np.load(trackingFile)
    tracking = data['tracking']
    fx = tracking[sf:ef,0] 
    fy = tracking[sf:ef,1]
    bx = tracking[sf:ef,2]
    by = tracking[sf:ef,3]
    ex = tracking[sf:ef,4]
    ey = tracking[sf:ef,5]
    area = tracking[sf:ef,6]
    ort = tracking[sf:ef,7]
    motion = tracking[sf:ef,8]
    return fx,fy,bx,by,ex,ey,area,ort,motion

def getDictsFromFolderList(f):
## Collects dictionary paths corresponding to paths generated using a folderListFile (only works if using specified folder structure)
    ROI_path,folderNames = read_folder_list(f)
    dictNameList=[]
    # Bulk analysis of all folders
    for idx,folder in enumerate(folderNames):
        AnalysisFolder,_ = get_analysis_folders(folder)
        
        dicSubFiles = glob.glob(AnalysisFolder + r'\*ANALYSIS.npy')
    # add to overall list (one by one)
    for s in dicSubFiles:dictNameList.append(s)
    return dictNameList
    
def getDictsFromTrackingFolder(f,suff=''):
## Collects dictionary paths corresponding to paths generated using a folder of tracking shortcuts (only works if using specified folder structure)
    shFiles=glob.glob(f+'\*.lnk')
    dictNameList=[]
    for i in range(len(shFiles)):
        ret,path=findShortcutTarget(shFiles[i])
        if(ret==0):
            d,_,f=path.rsplit(sep='\\',maxsplit=2)
            AnalysisFolder=d + '\\Analysis\\'
            f=f[0:-13]
            dicSubFiles = glob.glob(AnalysisFolder + r'\\*' + f + '*ANALYSIS' + suff + '.npy')
            for s in dicSubFiles:dictNameList.append(s)
        else:
            print('Could not find associated dictionary for ' + f)
            return -1
    return dictNameList

def getDictsFromTrackingFolderROI(file,anSuff='',suff=''):
## as previous function, but defining specific suffixes for analysis folder (anSuff) and experiment (suff). Can use to specify different Analysis rounds on the same experiment (for example when testing different analysis parameters)
    # cycle through the shortcuts and compile a list of targets
    shFiles=glob.glob(file+'\*.lnk')
    dictNameList=[]
    for i in range(len(shFiles)):
        ret,path=findShortcutTarget(shFiles[i])
        if(ret==0):
            d,_,f=path.rsplit(sep='\\',maxsplit=2)
            AnalysisFolder=d + '\\Analysis'+anSuff+'\\'
            f=f[0:-13]
            dicSubFiles = glob.glob(AnalysisFolder + r'\\*' + f + '*ANALYSIS_ROIs' + suff + '.npy')
            for s in dicSubFiles:dictNameList.append(s)
        else:
            print('Could not find associated dictionary for ' + f)
            return -1
    return dictNameList

def getDictsFromRootFolderROI(file,anSuff='',suff=''):
## as previous function, but defining specific suffixes for analysis folder (anSuff) and experiment (suff) and grabs from GroupedData folder on D:. Can use to specify different Analysis rounds on the same experiment (for example when testing different analysis parameters)
    # cycle through the shortcuts and compile a list of targets
    shFiles=glob.glob(file+'\*.lnk')
    dictNameList=[]
    for i in range(len(shFiles)):
        ret,path=findShortcutTarget(shFiles[i])
        if(ret==0):
            d,_,f=path.rsplit(sep='\\',maxsplit=2)
            AnalysisFolder='D:/Analysis' + suff + '/Dictionaries/'
            f=f[0:-13]
            dicSubFiles = glob.glob(AnalysisFolder + '/' + f + '*ANALYSIS_ROIs' + suff + '.npy')
            for s in dicSubFiles:dictNameList.append(s)
        else:
            print('Could not find associated dictionary for ' + f)
            return -1
    return dictNameList

def grabStimFileFromTrackingFile(path):
## grabs corresponding stimulus file path given the path to the tracking file (only works with specified file structure automatically generated by Step1)
    d,wDir,file=path.rsplit(sep='\\',maxsplit=2)
    file=file[0:-13]
    file=file+'.csv'
    string=d+r'\\StimFiles\\'+file
    return string

def grabStimFileFromTrackingFile1(path):
## grabs corresponding stimulus file path given the path to the tracking file (only works with specified file structure automatically generated by Step1)
    _,file=path.rsplit(sep='\\',maxsplit=1)
    file=file[0:-13]
    file=file+'.csv'
    string=r'D:\\StimFiles\\'+file
    return string
    
def grabAviFileFromTrackingFile(path):
## grabs corresponding avi file path given the path to the tracking file (only works with specified file structure automatically generated by Step1)
    
    d,wDir,file=path.rsplit(sep='\\',maxsplit=2)
    file=file[0:-13]
    file=file+'.avi'
    string=d+r'\\'+file
    return string

def grabTemplateFileFromTrackingFile1(path):
## grabs corresponding template file path given the path to the tracking file (only works with specified file structure automatically generated by Step1)
    
    _,file=path.rsplit(sep='\\',maxsplit=1)
    file=file[0:-13]
    string=r'D:\\Templates\\'+file+'_template.avi'
    return string

def grabTemplateFileFromTrackingFile(path):
## grabs corresponding template file path given the path to the tracking file (only works with specified file structure automatically generated by Step1)
    
    d,wDir,file=path.rsplit(sep='\\',maxsplit=2)
    file=file[0:-13]
    string=d+r'\\Templates\\'+file+'_template.avi'
    return string

def grabFishInfoFromFile(path):
## parses experiment name to find info (only works if specified naming convention used)
## INPUTS: path - full path (as a string) of avi file
## OUTPUTS:directory,name,date,gType,cond,chamber,fishNo.
    
    directory,file=path.rsplit(sep='\\',maxsplit=1)
    name,_=file.rsplit(sep=r'_',maxsplit=1)
    words=file[0:-4].split(sep=r'_')
    if len(words)<5:
        print('here')
    date=words[0]
    gType=words[1]
    cond=words[2]
    chamber=words[3]
    fishNo=words[4]
    
    return directory,name,date,gType,cond,chamber,fishNo

## Data handling/filtering ####################################################
    
def computeDist(x1,y1,x2,y2):
## Computes straight line distance between two points in space    
    absDiffX=np.abs(x1-x2)
    absDiffY=np.abs(y1-y2)
    dist = math.sqrt(np.square(absDiffX)+np.square(absDiffY))
    
    return dist

def computeDistPerBout(fx,fy,boutStarts,boutEnds):
## Computes total straight line distance travelled over the course of individual bouts
## Returns a distance travelled for each bout, and a cumDist    
    absDiffX=np.abs(fx[boutStarts]-fx[boutEnds])
    absDiffY=np.abs(fy[boutStarts]-fy[boutEnds])
    
    cumDistPerBout=np.zeros(len(boutStarts)-1)
    distPerBout=np.zeros(len(boutStarts))
    
    for i in range(len(boutStarts)):
        distPerBout[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
#        if distPerBout[i]>100:distPerBout[i]=0
        if i!=0 and i!=len(boutStarts)-1:
            cumDistPerBout[i]=distPerBout[i]+cumDistPerBout[i-1]
    
    return distPerBout,cumDistPerBout

def computeDistPerFrame(fx,fy):
## Computes straight line distance between every frame, given x and y coordinates of tracking data
    cumDistPerFrame=np.zeros(len(fx)-1)
    distPerFrame=np.zeros(len(fx))
    absDiffX=np.abs(np.diff(fx))
    absDiffY=np.abs(np.diff(fy))
    for i in range(len(fx)-1):
        if i!=0:
            distPerFrame[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
            if distPerFrame[i]>100:distPerFrame[i]=0
            cumDistPerFrame[i]=distPerFrame[i]+cumDistPerFrame[i-1]
    return distPerFrame,cumDistPerFrame

def checkTracking(distPerFrame,threshold=10):
# search tracking data for ridiculous jumps in distance (default is 10mm) and alert user if they exist     
    thresh=np.mean(distPerFrame)+(np.std(distPerFrame)*10)
    errorID=distPerFrame>thresh
    numErrorFrames=np.sum(errorID)
    percentError=((numErrorFrames/len(distPerFrame))*100)
    if(percentError>threshold):
        message=str(numErrorFrames) + r'or' + str(percentError) + ' of frames had unfeasible jumps in distance.'
        print(message)
    
def cropMotionFramesFromCumOrt(cumOrt,motion,preWindow=5,postWindow=25):
## Filters frames where the fish is moving from the cumulative orientation computation. The fish can be blurred during high motion, reducing reliability of the orientation computation. Instead, here it is inferred from frames either side where fish is not moving.
# dilate motion so that NOT moving is 1 and moving is -1
# take np.diff of cumOrt
# values of cumOrtDiff at motion before and after windows
    pol=postWindow+1
    dilatedMotion=np.zeros(len(motion))
    cumOrtDiff=np.diff(cumOrt)
    newcumOrtDiff=np.copy(cumOrtDiff)
    i=0
    for i in range(len(newcumOrtDiff)):
        if(motion[i]>0):
            if i > preWindow+1:
                pr=preWindow
                prl=preWindow+1
                pol=postWindow+1
            else: 
                pr=0
                prl=i
                pol=postWindow+1
            if i + pol > len(newcumOrtDiff): pol=i-len(newcumOrtDiff)
                
            newcumOrtDiff[i-pr:i]=cumOrtDiff[i-prl]
            newcumOrtDiff[i:i+postWindow]=cumOrtDiff[i+pol]
            
            i+=postWindow+preWindow
            if i > len(newcumOrtDiff)-1: break
        else:
            i+=1
            if i > len(newcumOrtDiff)-1: break
    
    dilatedMotion=dilatedMotion[0:len(newcumOrtDiff)]
    
    return newcumOrtDiff
    
def accumulate(x):
## Computes the cumulative vector of the input vector
    l=len(x)
    int_x=np.zeros(l)
    #x-=x[0]
    for i in range(l):
        if i!=0:
            int_x[i]=x[i]+int_x[i-1]
    return int_x

def convertToMm(XList,YList,pixwidth=0.09,pixheight=0.09): # pixel values based on measurement of entire chamber in pixels and mm from visual inspection through Bonsai. 100mm / 1100 pixels
## Converts the x and y pixel coordinates from tracking data into positions in mm. Can also be used to convert any list (or list of lists) from one unit to another given a scale factor for x and y
    
    XList_ret=[]
    YList_ret=[]
    for i,x in enumerate(XList):
        y=YList[i]
        XList_ret.append((x*pixwidth))
        YList_ret.append((y*pixheight))
    
    return XList_ret,YList_ret

def filterBursts(dispVec,frameRate=120,thresh=30,occludeWindow=10):
# filter "occludeWindow" seconds burst or escape activity (over "thresh") from dispersal vector, "dispVec"
    dispVecN=np.copy(dispVec)
    for i in range(0,len(dispVec)):
        if dispVec[i]>thresh:
            print(i)
            if i+(occludeWindow*frameRate)>len(dispVec):dispVecN[i-(occludeWindow*frameRate):-1]=np.nan
            else:dispVecN[i-(occludeWindow*frameRate):i+(occludeWindow*frameRate)]=np.nan
    return dispVecN

## Plotting for testing #######################################################
    
def plotMotionMetrics(trackingFile,startFrame,endFrame):
## plots tracking trajectory, motion, distance per frame and cumulative distance for defined section of tracking data
    
    fx,fy,bx,by,ex,ey,area,ort,motion=load_trackingFile(trackingFile)
    plt.figure()
    plt.plot(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.title('Tracking')
    
    smoothedMotion=AZM.smoothSignal(motion[startFrame:endFrame],120)
    plt.figure()
    plt.plot(smoothedMotion)
    plt.title('Smoothed Motion')
    
    distPerFrame,cumDistPerFrame=computeDistPerFrame(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.figure()
    plt.plot(distPerFrame)
    plt.title('Distance per Frame')
    
    xx=AZM.smoothSignal(distPerFrame,30)
    plt.figure()
    plt.plot(xx[startFrame:endFrame])
    plt.title('Smoothed Distance per Frame (30 seconds)')
    
    
    plt.figure()
    plt.plot(cumDistPerFrame)
    plt.title('Cumulative distance')    
    
    return cumDistPerFrame

def trackFrame(aviFile,f0,f1,divisor):
## Tracks fish across two defined frames (f0 and f1) of a given movie (aviFile) and using a background threshold divided by defined divisor.
## Returns a threshold value for this tracking iteration for this movie that depends on given divisor    
## Used to test background computation parameters and diagnosis of problem frames in tracking  
    
    vid = cv2.VideoCapture(aviFile)
    im0=grabFrame(vid,f0)
    im1=grabFrame(vid,f1)
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    diff = im0 - im1
    threshold_level = np.median(im0)/divisor
    level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
    threshold = np.uint8(threshold)
    #threshold = np.uint8(diff > threshold_level)
    return threshold

def trackFrameBG(ROI,aviFile,f1,divisor):
## Tracks fish across a defined frames (f1) compared to the computed background of a given movie (aviFile) and using a background threshold divided by defined divisor.
## Returns a threshold value for this tracking iteration for this movie that depends on given divisor    
## Used to test background computation parameters and diagnosis of problem frames in tracking  
    vid = cv2.VideoCapture(aviFile)
    im0=AZVJ.compute_initial_background(aviFile, ROI)
    im1=grabFrame(vid,f1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    diff = im0 - im1
    threshold_level = np.median(im0)/divisor
    level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
    threshold = np.uint8(threshold)
    #threshold = np.uint8(diff > threshold_level)
    return threshold

## Video handling #############################################################    
def cropMovie(aviFile,ROI,outname='Cropped.avi',FPS=100):
    
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = AZVJ.get_ROI_sizeSingle(ROI)
    outFile=r'D:\\Movies\\cache\\'+outname
    out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (w,h), False)
    for i in range(numFrames):
        im=np.uint8(grabFrame32(vid,i))
        crop, _,_ = AZVJ.get_ROI_cropSingle(im,ROI)
        out.write(crop)
    out.release()
    
    return 0

def trimMovie(aviFile,startFrame,endFrame,saveName):
## Creates a new movie file with 'saveName' and desired start and end frames.    
## INPUTS:  aviFile - string with full path of aviFile
##          startFrame,endFrame - the desired start and end positions of new movie
##          saveName - string with full path of new save location
     
    FPS=120
    vid=cv2.VideoCapture(aviFile)
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if(endFrame==-1) or (endFrame>numFrames):
        endFrame=numFrames
        
    out = cv2.VideoWriter(saveName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
    setFrame(vid,startFrame)
    
    for i in range(endFrame-startFrame):
        ret, im = vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        out.write(im)
        
    out.release()
    vid.release()
    return 0

def setFrame(vid,frame):
## set frame of a cv2 loaded movie without having to type the crazy long cv2 command
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    return 0

def grabFrame(avi,frame):
# grab frame and return the image from loaded cv2 movie
    vid=cv2.VideoCapture(avi)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid.release()
    im = np.uint8(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im

def grabFrame32(vid,frame):
# grab frame and return the image (float32) from loaded cv2 movie
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im
   
def showFrame(vid,frame):
# display selected frame (greyscale) of a cv2 loaded movie (for testing)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    plt.figure()
    plt.imshow(im)
    return 0

def read_folder_list(folderListFile): 
## Read Folder List file 
    folderFile = open(folderListFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines
    data_path = folderList[0][:-1] # Read Data Path which is the first line
    folderList = folderList[1:] # Remove first line becasue it contains the path
    
    folderNames = [] # We use this becasue we do not know the exact length
    
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
        stringLine = f[:].split()
        expFolderName = data_path + stringLine[0]
        folderNames.append(expFolderName)
        
    return data_path,folderNames
    

def get_analysis_folders(folder):
## Determine Analysis Folder Names from Root directory
    # Specifiy Folder Names
    AnalysisFolder = folder + 'Analysis'
    TemplateFolder = folder + 'Templates'
    TrackingFolder = folder + 'Tracking'
     
    return AnalysisFolder, TemplateFolder, TrackingFolder

def load_trackingFile(filename):
## Duplicated...?
    data = np.load(filename)
    tracking = data['tracking']
        
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]
    
    return fx,fy,bx,by,ex,ey,area,ort,motion

### Testing and tuning tracking
# Quantify Tracking Data (remove errors in tracking)
def measure_tracking_errors(tracking):
    X = tracking[:, 0]
    Y = tracking[:, 1]
#    Ort = tracking[:, 2]
#    MajAx = tracking[:, 3]
#    MinAx = tracking[:, 4]
    Area = tracking[:, 5]
    
    # Filter out and interpolate between tracking errors
    speedXY =  compute_speed(X,Y)
    tooFastorSmall = (speedXY > 50) + (Area < 75)    
    trackingErrors = np.sum(tooFastorSmall)    
    
    return trackingErrors


# Quantify Tracking Data (remove errors in tracking)
def burst_triggered_alignment(starts, variable, offset, length):
    starts = starts[starts > offset]
    numStarts = np.size(starts)
    aligned = np.zeros((numStarts, length))

    for s in range(0, numStarts):
        aligned[s, :] = variable[starts[s]-offset:starts[s]-offset+length]

    return aligned# Peak Detection
def find_peaks(values, threshold, refract):    
    over = 0
    r = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[2:-100]:
        if over == 0:
            if values[i] > threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i-1)
        else: #This is what happens when over the threshold
            if r < refract:
                r = r + 1
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
            else:
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
                elif values[i] < threshold:
                    over = 0
                    r = 0
                    curPeakVal = 0
                    peaks.append(curPeakIdx)
                    stops.append(i)
    
    return starts, peaks, stops

# Peak Detection
def find_peaks_dual_threshold(values, upper_threshold, lower_threshold):    
    over = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[5:-100]:
        if over == 0:
            if values[i] > upper_threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i)
        else: #This is what happens when over the upper_threshold
            if values[i] > curPeakVal:
                curPeakVal = values[i]
                curPeakIdx = i
            elif values[i] < lower_threshold:
                over = 0
                curPeakVal = 0
                peaks.append(curPeakIdx)
                stops.append(i)
    
    return starts, peaks, stops

def diffAngle(Ort):
## Computes the change in angle over all frames of given Ort tracking data
    dAngle = np.diff(Ort)
    new_dAngle = [0]    
    for a in dAngle:
        if a < -270:
            new_dAngle.append(a + 360)
        elif a > 270:
            new_dAngle.append(a - 360)
        else:
            new_dAngle.append(a)
    
    return np.array(new_dAngle)

def filterTrackingFlips(dAngle):
## Identifies and reverses sudden flips in orientation caused by errors in tracking the eyes vs the body resulting in very high frequency tracking flips    
    new_dAngle = []    
    for a in dAngle:
        if a < -100:
            new_dAngle.append(a + 180)
        elif a > 100:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)

def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed

def motion_signal(X, Y, Ort):
## Combines velocity and angular velocity each weighted by their standard deviation to give a combined 'motion_signal' metric of movement. 
    SpeedXY, SpeedAngle=compute_bout_signals(X, Y, Ort)
    
    # Absolute Value of angular speed
    SpeedAngle = np.abs(SpeedAngle)

    # Weight contribution by STD
    std_XY = np.std(SpeedXY)    
    std_Angle = np.std(SpeedAngle)    
    SpeedXY = SpeedXY/std_XY
    SpeedAngle = SpeedAngle/std_Angle

    # Sum Combined Signal
    motion_signal = SpeedXY+SpeedAngle

    return SpeedXY,motion_signal

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals(X, Y, Ort):

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
#    # Filter Speed for outliers
    sigma = np.std(speedXY)
    baseline = np.median(speedXY)
    speedXY[speedXY > baseline+10*sigma] = -1.0
    
    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
    
    speedAngle = filterTrackingFlips(speedAngle)
    
    return speedXY, speedAngle

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals_calibrated(X, Y, Ort, ROI, test):
    
    # Calibrate X and Y in ROI units
    offX = ROI[0]
    offY = ROI[1]
    width = ROI[2]
    height = ROI[3] 
    X = (X - offX)/width
    Y = (Y - offY)/height
    if test:
        X = X * 14; # Convert to mm
        Y = Y * 42; # Convert to mm
    else:
        X = X * 14; # Convert to mm
        Y = Y * 14; # Convert to mm
        

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
#    # Filter Speed for outliers
#    sigma = np.std(speedXY)
#    baseline = np.median(speedXY)
#    speedXY[speedXY > baseline+10*sigma] = -1.0

    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
    
    return speedXY, speedAngle

# Extract Bouts from Motion Signal
def extract_bouts_from_motion(X, Y, Ort, motion, upper_threshold, lower_threshold, ROI, test):

    if test:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, True)
    else:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, False)        
     
    # Find Peaks in Motion Signal 
    starts, peaks, stops = find_peaks_dual_threshold(motion, upper_threshold, lower_threshold)
    numBouts = np.size(peaks)    
    bouts = np.zeros([numBouts, 6])

    for i in range(numBouts):
        bouts[i, 0] = starts[i]-4 # Point 4 frames (40 ms) before Upper threshold crossing 
        bouts[i, 1] = peaks[i] # Peak
        bouts[i, 2] = stops[i]+1 # Point 1 frame (10 ms) after lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(SpeedAngle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sum(SpeedXY[starts[i]:stops[i]]) # Net distance change

    return bouts

def polar_orientation(Ort):
## Generates a polar histogram of the orientation of the fish
    ort_hist, edges = np.histogram(Ort, 18, (0, 360))
    plt.plot(edges/(360/(2*np.pi)), np.append(ort_hist, ort_hist[0]))
    max_ort = edges[np.argmax(ort_hist)]
    return max_ort

#filepath='D:\\\\StimFiles\\\\210325_EmxGFP_Ctrl_B2_1.csv'
#loomStarts,loomEnds,respEnds,loomPosX,loomPosY=findLoomsFromFile(filepath)
# FIN