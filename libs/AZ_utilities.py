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
import AZ_video as AZV
import os
import numpy as np
import matplotlib.pyplot as plt
import CV_ARK
import cv2
from win32com.client import Dispatch
import glob

def getDictsFromFolderList(f):
    
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
    # cycle through the shortcuts and compile a list of targets
    shFiles=glob.glob(f+'\*.lnk')
    dictNameList=[]
    for i in range(len(shFiles)):
        ret,path=findShortcutTarget(shFiles[i])
        if(ret==0):
            d,_,f=path.rsplit(sep='\\',maxsplit=2)
            AnalysisFolder=d + '\\AnalysisBoutSeqTest\\'
            f=f[0:-13]
            dicSubFiles = glob.glob(AnalysisFolder + r'\\*' + f + '*ANALYSIS' + suff + '.npy')
            for s in dicSubFiles:dictNameList.append(s)
        else:
            print('Could not find associated dictionary for ' + f)
            return -1
    return dictNameList

def getDictsFromTrackingFolderROI(file,anSuff='',suff=''):
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

def grabTrackingFromFile(trackingFile,sf=0,ef=-1):
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

def cropMotionFramesFromCumOrt(cumOrt,motion,preWindow=5,postWindow=25):
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
    l=len(x)
    int_x=np.zeros(l)
    #x-=x[0]
    for i in range(l):
        if i!=0:
            int_x[i]=x[i]+int_x[i-1]
    return int_x
#-----------------------------------------------------------------------------
def convertToMm(XList,YList,pixwidth=0.09,pixheight=0.09): # pixel values based on measurement of entire chamber in pixels and mm from visual inspection through Bonsai. 100mm / 1100 pixels
    
    XList_ret=[]
    YList_ret=[]
    for i,x in enumerate(XList):
        y=YList[i]
        XList_ret.append((x*pixwidth))
        YList_ret.append((y*pixheight))
    
    return XList_ret,YList_ret

def grabAviFileFromTrackingFile(path):
    
    d,wDir,file=path.rsplit(sep='\\',maxsplit=2)
    file=file[0:-13]
    file=file+'.avi'
    string=d+r'\\'+file
    return string

def grabFishInfoFromFile(path):
    
    directory,_,file=path.rsplit(sep='\\',maxsplit=2)
    name,_=file.rsplit(sep=r'_',maxsplit=1)
    words=file[0:-4].split(sep=r'_')
    
    date=words[0]
    gType=words[1]
    cond=words[2]
    chamber=words[3]
    fishNo=words[4]
    
    return directory,name,date,gType,cond,chamber,fishNo
    
def findShortcutTarget(path):
    
    shell = Dispatch("WScript.Shell")
    try:
        shortcut = shell.CreateShortCut(path)
    except OSError:
        return -1,"_"
    else:
        return 0,shortcut.Targetpath

def trackingSwitcher(i,types=["tracking","cropped","initial_background","final_background","initial_tracking","final_tracking"]):
    
    for k in range(len(types)):
        switcher={k:types[k]}
    
    return switcher.get(i,-1)

def createShortcut(target,location):
    
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
                
                
def createShortcutTele(target,root=[],location="default",types=["tracking","cropped","initial_background","final_background","initial_tracking","final_tracking","avgBout","heatmap","cumDist","boutAmpsHist"]):

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
    
    
## Create a new movie file with 'saveName' and desired start and end frames.    
def trimMovie(aviFile,startFrame,endFrame,saveName):
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
    
# filter "occludeWindow" seconds burst or escape activity (over "thresh") from dispersal vector, "dispVec"
def filterBursts(dispVec,frameRate=120,thresh=30,occludeWindow=10):
    dispVecN=np.copy(dispVec)
    for i in range(0,len(dispVec)):
        if dispVec[i]>thresh:
            print(i)
            if i+(occludeWindow*frameRate)>len(dispVec):dispVecN[i-(occludeWindow*frameRate):-1]=np.nan
            else:dispVecN[i-(occludeWindow*frameRate):i+(occludeWindow*frameRate)]=np.nan
    return dispVecN

def plotMotionMetrics(trackingFile,startFrame,endFrame):
    
    fx,fy,bx,by,ex,ey,area,ort,motion=load_trackingFile(trackingFile)
    plt.figure()
    plt.plot(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.title('Tracking')
    
    smoothedMotion=smoothSignal(motion[startFrame:endFrame],120)
    plt.figure()
    plt.plot(smoothedMotion)
    plt.title('Smoothed Motion')
    
    distPerFrame,cumDistPerFrame=computeDistPerFrame(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.figure()
    plt.plot(distPerFrame)
    plt.title('Distance per Frame')
    
    xx=smoothSignal(distPerFrame,30)
    plt.figure()
    plt.plot(xx[startFrame:endFrame])
    plt.title('Smoothed Distance per Frame (30 seconds)')
    
    
    plt.figure()
    plt.plot(cumDistPerFrame)
    plt.title('Cumulative distance')    
    
    return cumDistPerFrame
    
def smoothSignal(x,N):

    xx=np.convolve(x, np.ones((int(N),))/int(N), mode='valid')
    n=N-1
    xpre=np.zeros(n)
    xxx=np.concatenate((xpre,xx))
    return xxx

def trackFrame(aviFile,f0,f1,divisor):
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

def trackFrameBG(ROI,aviFileB,aviFile,f0,f1,divisor):
    vid = cv2.VideoCapture(aviFile)
    im0=AZV.compute_initial_background(aviFile, ROI)
    im1=grabFrame(vid,f1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    diff = im0 - im1
    threshold_level = np.median(im0)/divisor
    level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
    threshold = np.uint8(threshold)
    #threshold = np.uint8(diff > threshold_level)
    return threshold

# set frame without having to type the crazy long cv2 command
def setFrame(vid,frame):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)


# grab frame and return the image (float32)
def grabFrame32(vid,frame):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im
   # grab frame and return the image
def grabFrame(avi,frame):
    vid=cv2.VideoCapture(avi)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid.release()
    im = np.uint8(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im
# shows selected frame of a cv2 object/ video (for testing)
def showFrame(vid,frame):

    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    plt.figure()
    plt.imshow(im)

def checkTracking(distPerFrame,threshold=10):
     
    thresh=np.mean(distPerFrame)+(np.std(distPerFrame)*10)
    errorID=distPerFrame>thresh
    numErrorFrames=np.sum(errorID)
    percentError=((numErrorFrames/len(distPerFrame))*100)
    if(percentError>threshold):
        message=str(numErrorFrames) + r'or' + str(percentError) + ' of frames had unfeasible jumps in distance.'
        print(message)
    
    
def computeDist(x1,y1,x2,y2):
    
    absDiffX=np.abs(x1-x2)
    absDiffY=np.abs(y1-y2)
    dist = math.sqrt(np.square(absDiffX)+np.square(absDiffY))
    
    return dist

    
def computeDistPerBout(fx,fy,boutStarts,boutEnds):
    
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
    
# 1) Mkdir with error reporting

def tryMkDir(path,report=0):

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
        
def cycleMkDir(path,report=0):
    
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
    
    splitPath=path.split(sep="\\")
    for i,name in enumerate(splitPath):
        if(i==0):
            s=name+"\\"
        else:
            s=s+name+"\\"
        if(i<len(splitPath)-1):    
            tryMkDir(s,report=report)
        
        
    
# 2) Read Folder List file 
def read_folder_list(folderListFile): 
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
    
# 2) Determine Analysis Folder Names from Root directory
def get_analysis_folders(folder):
    # Specifiy Folder Names
    AnalysisFolder = folder + 'Analysis'
    TemplateFolder = folder + 'Templates'
    TrackingFolder = folder + 'Tracking'
     
    return AnalysisFolder, TemplateFolder, TrackingFolder

    
# Read ROIs (expecting 12)
def read_crop_ROIs(roiFilename):
    ROIs = CV_ARK.read_roi_zip(roiFilename)
    numROIs = len(ROIs)
    if (numROIs != 12):
        raise ValueError('roireader: Expecting 12 social ROIs, found %i' % numROIs)
    test_ROIs = np.zeros((6, 4))
    stim_ROIs = np.zeros((6, 4))
    
    test_ROIs[0,:] = np.array(ROIs[0])
    test_ROIs[1,:] = np.array(ROIs[3])
    test_ROIs[2,:] = np.array(ROIs[4])
    test_ROIs[3,:] = np.array(ROIs[7])
    test_ROIs[4,:] = np.array(ROIs[8])
    test_ROIs[5,:] = np.array(ROIs[11])
    
    stim_ROIs[0,:] = np.array(ROIs[1])
    stim_ROIs[1,:] = np.array(ROIs[2])
    stim_ROIs[2,:] = np.array(ROIs[5])
    stim_ROIs[3,:] = np.array(ROIs[6])
    stim_ROIs[4,:] = np.array(ROIs[9])
    stim_ROIs[5,:] = np.array(ROIs[10])
    
    return test_ROIs, stim_ROIs

def load_trackingFile(filename):
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

# Load Tracking Data
def load_tracking(filename):

    # Read Column Headers
    datafile = open(filename, 'r')
    header = datafile.readline()
    header = header[:-1]                # Remove newline char
    column_lables = header.split(' ')

    # Read data from file
    data = np.genfromtxt(filename, delimiter=' ', skiprows=1)

    # Allocate space and fill data into appropriate columns
    tracking = np.zeros(np.shape(data))    
    tracking[:, 0] = data[:, column_lables.index('Centroid.X')]     
    tracking[:, 1] = data[:, column_lables.index('Centroid.Y')]     
    tracking[:, 2] = data[:, column_lables.index('Orientation')]     
    tracking[:, 3] = data[:, column_lables.index('MajorAxisLength')]     
    tracking[:, 4] = data[:, column_lables.index('MinorAxisLength')]     
    tracking[:, 5] = data[:, column_lables.index('Area')]
    
    return tracking

# Load Tracking Data
def plot_spatial_variable(X,Y,Var):
    
    width = np.ceil(np.max(X)-np.min(X))+1
    height = np.ceil(np.max(Y)-np.min(Y))+1
    X = X - np.min(X)
    Y = Y - np.min(Y)
    
    space = np.zeros([height, width])    
    data = np.vstack((X,Y,Var))
    data = data.T

    for x,y,var in data:
        space[np.round(y),np.round(x)] = var
    
    plt.imshow(space)

# Peak Detection
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

# Adjust Orientation (Test Fish)
def adjust_ort_test(ort, chamber):
    # Adjust orientations so 0 is always pointing towards "other" fish
    if chamber%2 == 0: # Test Fish facing Left
        for i,angle in enumerate(ort):
            if angle >= 0: 
                ort[i] = angle - 180
            else:
                ort[i] = angle + 180
    return ort

# Adjust Orientation (Stim Fish)
def adjust_ort_stim(ort, chamber):
    # Adjust orientations so 0 is always pointing towards "other" fish
    if chamber%2 == 1: # Stim Fish facing Left
        for i,angle in enumerate(ort):
            if angle >= 0: 
                ort[i] = angle - 180
            else:
                ort[i] = angle + 180
    return ort

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


# Make Polar Plot of Orientation
def polar_orientation(Ort):
    ort_hist, edges = np.histogram(Ort, 18, (0, 360))
    plt.plot(edges/(360/(2*np.pi)), np.append(ort_hist, ort_hist[0]))
    max_ort = edges[np.argmax(ort_hist)]
    return max_ort

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

    return aligned

# Find nearest index in array 2 from array 1
def find_dist_to_nearest_index(array1, array2):
    numIndices = np.size(array1)    
    distances = np.zeros(np.shape(array1))
    for i in range(0, numIndices):
        diffArray = array2 - array1[i]
        closestIndex = np.argmin(np.abs(diffArray))
        distances[i] = diffArray[closestIndex]
    
    return distances
    
# Find nearest index in array 2 AFTER those in array 1
def find_next_nearest_index(array1, array2):
    numIndices = np.size(array1)    
    nextIndices = np.zeros(np.shape(array1))
    for i in range(0, numIndices):
        diffArray = array2 - array1[i]
        positive = np.where(diffArray > 0)[0]
        if np.size(positive) != 0:
            nextIndex = positive[0]
            nextIndices[i] = diffArray[nextIndex]
    
    return nextIndices



    
def get_folder_names_controls(folder):
    # Specifiy Folder Names
    NS_folder = folder + r'/Non_Social_1'
    
    S_folder = folder + r'/Non_Social_2'
    if os.path.exists(S_folder) == False:
        S_folder = folder + r'/Social_1_Real'
        
#    D_folder = folder + r'/Social_Dark'
#    if os.path.exists(D_folder) == False:
#        D_folder = -1
#    
#    C_folder = folder + r'/Non_Social_2'
#    if os.path.exists(C_folder) == False:
#        C_folder = -1    
    
    return NS_folder, S_folder
    
def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

     
# FIN