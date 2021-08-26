# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:05:25 2021

@author: thoma
"""
LIBROOT = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish'

# Set library paths
import sys
lib_path = LIBROOT + "/ARK/libs"
ARK_lib_path = LIBROOT + "/libs"
sys.path.append(lib_path)
sys.path.append(ARK_lib_path)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import AZ_utilities as AZU

def plotRNNMeasure(subplotNumS,title,pred,predShuff,yTest,yTestShuff,ss=2,legend=False,MetLab=''):
    plt.subplot(subplotNumS[0],subplotNumS[1],subplotNumS[2])
    plt.title(title)
    plt.scatter(predShuff,yTest,alpha=0.3,color='red',s=ss,label='Shuffled')
    plt.scatter(pred,yTest,alpha=0.3,color='black',s=ss,label='True')
    plt.xlabel('Predicted ' + MetLab)
    plt.ylabel('Actual ' + MetLab)
    if legend:
        plt.legend(loc='upper right')        
        
def plotRNNMeasureSepFigs(title,pred,predShuff,yTest,yTestShuff,textstr,saveDir='',saveFig=False,keepFigures=True,ss=2,legend=False,MetLab=''):
    plt.figure()
    plt.title(title)
    plt.scatter(predShuff,yTest,alpha=0.3,color='red',s=ss,label='Shuffled')
    plt.scatter(pred,yTest,alpha=0.3,color='black',s=ss,label='True')
    plt.xlabel('Predicted ' + MetLab)
    plt.ylabel('Actual ' + MetLab)
    if legend:
        plt.legend(loc='upper right')        
    ax=plt.gca()
    txtXloc=((np.max(pred)-np.min(pred))*0.1)+np.min(pred)
    txtYloc=((np.max(yTest)-np.min(yTest))*0.7)+np.min(yTest)
    ax.text(txtXloc, txtYloc, textstr, fontsize=10)
    figname=title + '_Summary.png'
    if saveFig:
        AZU.cycleMkDir(saveDir)
        saveName = saveDir + figname
        plt.savefig(saveName,dpi=600)
    if keepFigures==False:
        plt.close()
        
def testModel(model,xTest,yTest,Uni=True,hack=True):
    
    if Uni:
        if hack:
            xTest=makeSeries(xTest)
        else:
            yTest=yTest[:,0,0]
        xTestShuff=xTest[np.random.permutation(np.arange(0,xTest.shape[0],1))]
        yTestShuff=np.random.permutation(yTest)        
        pred=model.predict(xTest)[:,0]
        predShuff=model.predict(xTestShuff)[:,0]
        mae_trueS=np.mean(np.abs(pred-yTest))
        mae_shuff_pS=np.mean(np.abs(predShuff-yTest))
        mae_shuff_tS=np.mean(np.abs(pred-yTestShuff))
    else:
        xTestShuff=xTest[np.random.permutation(np.arange(0,xTest.shape[0],1)),:,:]
        yTestShuff=yTest[np.random.permutation(np.arange(0,yTest.shape[0],1)),:]
        
        pred=model.predict(xTest)
        predShuff=model.predict(xTestShuff) # real preictions on shuffled inputs
        
        mae_trueS,mae_shuff_tS,mae_shuff_pS=[],[],[]
        
        for i in range(0,xTest.shape[2]):
            thisYTest=yTest[:,i]
            thisYTestShuff=yTestShuff[:,i]
            predtemp=pred[:,i]
            predShufftemp=predShuff[:,i]
            mae_trueS.append(np.mean(np.abs(predtemp-thisYTest)))
            mae_shuff_pS.append(np.mean(np.abs(predShufftemp-thisYTest)))
            mae_shuff_tS.append(np.mean(np.abs(predtemp-thisYTestShuff)))
        
    return pred,predShuff,yTestShuff,mae_trueS,mae_shuff_pS,mae_shuff_tS

#def clipAndStandardise(DATALIST,paramLabels,scaleMethod='max',scale=True,THRESHOLDS=[100,180,60]):
#    
#    for i, data in enumerate(DATALIST):
#        for j,paramLabel in enumerate(paramLabels):
#            tr=data[:,:-1,j]
#            t=data[:,-1,j]
#            if j == 0:
#                tr[tr>THRESHOLDS[0]]=THRESHOLDS[0]
#                t[t>THRESHOLDS[0]]=THRESHOLDS[0] # hack to get rid of very high values (where are these from!?)
#            elif j==1:
#                tr[tr>THRESHOLDS[1]]=THRESHOLDS[1] # hack to get rid of very high values (where are these from!?)
#                tr[tr<-(THRESHOLDS[1]*-1)]=(THRESHOLDS[1]*-1)
#                t[t>THRESHOLDS[1]]=THRESHOLDS[1] # hack to get rid of very high values (where are these from!?)
#                t[t<(THRESHOLDS[1]*-1)]=(THRESHOLDS[1]*-1)
#            elif j==2:
#                tr[tr>THRESHOLDS[2]]=THRESHOLDS[2] # hack to get rid of very high values (where are these from!?)
#                t[t>THRESHOLDS[2]]=THRESHOLDS[2]
#            if scale:
#                # Distance and IBI are log distributed
#                # log then normalise (zero mean, min/max)
#                # angle is normally distributed around 0
#                # normalise to max (180)
#                if j == 0 or j==2:
#                    DATALIST[i][:,:-1,j]=logStandardiseRNN(tr,method=scaleMethod,logg=True)
#                elif j ==1:
#                    DATALIST[i][:,:-1,j]=logStandardiseRNN(tr,method=scaleMethod,logg=False)
#            else:
#                DATALIST[i][:,:-1,j]=tr
#            DATALIST[i][:,-1,j]=t
#    return DATALIST

def clipAndStandardise(DATALIST,paramLabels,scaleTarget=False,scaleMethod='max',scale=True,THRESHOLDS=[100,180,60]):
    
    for i, data in enumerate(DATALIST):
        for j,paramLabel in enumerate(paramLabels):
            t=data[:,-1,j]
            if scaleTarget:
                tr=data[:,:,j]
            else:
                tr=data[:,:-1,j]
                
            if j == 0:
                tr[tr>THRESHOLDS[0]]=THRESHOLDS[0]
                t[t>THRESHOLDS[0]]=THRESHOLDS[0] # hack to get rid of very high values (where are these from!?)
            elif j==1:
                tr[tr>THRESHOLDS[1]]=THRESHOLDS[1] # hack to get rid of very high values (where are these from!?)
                tr[tr<(THRESHOLDS[1]*-1)]=(THRESHOLDS[1]*-1)
                t[t>THRESHOLDS[1]]=THRESHOLDS[1] # hack to get rid of very high values (where are these from!?)
                t[t<(THRESHOLDS[1]*-1)]=(THRESHOLDS[1]*-1)
            elif j==2:
                tr[tr>THRESHOLDS[2]]=THRESHOLDS[2] # hack to get rid of very high values (where are these from!?)
                t[t>THRESHOLDS[2]]=THRESHOLDS[2]
            if scale:
                # Distance and IBI are log distributed
                # log then normalise (zero mean, min/max)
                # angle is normally distributed around 0
                # normalise to max (180)
                if j == 0 or j==2:
                    if scaleTarget: # then scale the whole dataset
                        DATALIST[i][:,:,j]=logStandardiseRNN(tr,method=scaleMethod,logg=True)
                    else: # otherwise only scale the inputs, not the targets (:-1 of each series)
                        DATALIST[i][:,:-1,j]=logStandardiseRNN(tr,method=scaleMethod,logg=True)
                        DATALIST[i][:,-1,j]=t
                elif j ==1: 
                    if scaleTarget: # then scale the whole dataset
                        DATALIST[i][:,:,j]=logStandardiseRNN(tr,method=scaleMethod,logg=False)
                    else: # otherwise only scale the inputs, not the targets (:-1 of each series)
                        DATALIST[i][:,:-1,j]=logStandardiseRNN(tr,method=scaleMethod,logg=False)
                        DATALIST[i][:,-1,j]=t
            else:
                DATALIST[i][:,:-1,j]=tr
                DATALIST[i][:,-1,j]=t
                
    return DATALIST

def generateSinWaveSeries(batch_size,n_steps,freq=5,Split=False,training_split=0.7,addNoise=True):
    offsets=np.random.rand(batch_size,1)
    
    if Split:
        n_steps+=1
    time=np.linspace(0,1,n_steps)
    series=np.sin((time-offsets)*(freq *10+10))
    if addNoise:
        series+=0.1*(np.random.rand(batch_size,n_steps)-0.5)
    if Split:
        num_train=np.int(batch_size*training_split)
        xTrain,yTrain = series[:num_train,:n_steps-1],series[:num_train,-1]
        xTest,yTest = series[num_train:,:n_steps-1],series[num_train:,-1]
        return [xTrain,yTrain,xTest,yTest]
    else:
        return series[...,np.newaxis].astype(np.float32)


def build_and_compile_RNNmodel(nNeurons,loss,opt,met,scaleTarget=False,LSTM=False,input_shape=(None,3)):
    
    if LSTM:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(nNeurons[0],return_sequences=True,input_shape=input_shape))
        for thisLayer in nNeurons[1:-2]:
            model.add(tf.keras.layers.LSTM(thisLayer,return_sequences=True))
        if scaleTarget:
            model.add(tf.keras.layers.LSTM(nNeurons[-2],return_sequences=True))
            model.add(tf.keras.layers.LSTM(nNeurons[-1]))
        else:
            model.add(tf.keras.layers.LSTM(nNeurons[-2],return_sequences=False))
            model.add(tf.keras.layers.Dense(nNeurons[-1]))
    else:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(nNeurons[0],return_sequences=True,input_shape=input_shape))
        for thisLayer in nNeurons[1:-2]:
            model.add(tf.keras.layers.SimpleRNN(thisLayer,return_sequences=True))
        if scaleTarget:
            model.add(tf.keras.layers.SimpleRNN(nNeurons[-2],return_sequences=True))
            model.add(tf.keras.layers.SimpleRNN(nNeurons[-1]))
        else:
            model.add(tf.keras.layers.SimpleRNN(nNeurons[-2],return_sequences=False))
            model.add(tf.keras.layers.Dense(nNeurons[-1]))
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[met])
    return model

def build_and_compile_DNNmodel(nNeurons,loss,opt,met,input_shape=(None,3)):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nNeurons[0],input_shape=input_shape))
    for thisLayer in nNeurons[1:]:
        model.add(tf.keras.layers.Dense(thisLayer))
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[met])
    return model

def logStandardiseRNN(Series,method='max',logg=True):
    
    if logg:
        minn=np.min(Series)
        Series-= minn
        Series = np.log(Series+1)
        Series+= minn
        
    if method=='max':# then normalise (min/max, 0:1)
        Series-= np.min(Series)
        Series/=np.max(Series)
    elif method=='std':# then standardise (zero mean unit variance)
        Series-=np.mean(Series)
        Series/=np.std(Series)
    elif method=='negMax':# then normalise (min/max -1:1)
        Series-=np.mean(Series)
        Series/=np.max(np.abs(Series))
    return Series

def makeSeries(vec):
    if len(vec.shape)==2:
        series=np.zeros([vec.shape[0],vec.shape[1],1])
        series[:,:,0]=vec
    if len(vec.shape)==1:
        series=np.zeros([vec.shape[0],1,1])
        for i,iS in enumerate(vec):
            series[i][0][0]=iS
    return series

def makeSeriesRNN(vec):
    if len(vec.shape)==2:
        series=np.zeros([vec.shape[0],vec.shape[1],1])
        for i in range(vec.shape[1]):
            for j,val in enumerate(vec[:,i]):
                series[:,:,i]=val
                
    if len(vec.shape)==1:
        series=np.zeros([vec.shape[0],1,1])
        for i,iS in enumerate(vec):
            series[i][0][0]=iS
    return series
#### 
def countAllBoutsSplit(groups,history_length,removeClippedBouts=False,omitForwards=False,turnThresh=9.95,train_split=0.7,unroll=False):
    num_train,num_test=0,0
    train_indicesS,test_indicesS=[],[]
    for i,group in enumerate(groups):
        for j,path in enumerate(group):
            bouts=np.load(path)['bouts']
            if omitForwards:
                dAngle=bouts[0:-1,4]
                num_bouts=np.int(len(dAngle[np.abs(dAngle)>turnThresh])-2)
            else:
                num_bouts=np.int(len(bouts[0:-1,4])-2)
                
            num_bouts=num_bouts-history_length
            num_train_thisFish=np.int(num_bouts*train_split)
            num_train+=num_train_thisFish
            num_test+=np.int(num_bouts-num_train_thisFish)
            if unroll:
                train_indicesS.append(range(0,num_train_thisFish))
                test_indicesS.append(range(num_train_thisFish,num_bouts))
            else:
                test_indicesS.append(np.random.randint(num_train_thisFish, high=num_bouts, size=np.int(num_bouts-num_train_thisFish)))
                train_indicesS.append(np.random.randint(0,high=num_train_thisFish , size=num_train_thisFish))
                                           
    return num_train,num_test,train_indicesS,test_indicesS

def buildTrainTestSetsRNN(groups,removeClippedBouts=False,omitForwards=False,turnThresh=9.95,keepGroupsSeperate=False,num_param=3,shuffle=False,FPS=120,history_length=5,prediction_target=1,train_split=0.7):
    
    if len(groups)>4:groups=[groups] # if more than 4, then is unlikely to be a list of lists
    num_fish=0
    for i in groups:
        num_fish+=len(i)
        
    # Create train / test datasets
    num_train,num_test,train_indicesS,test_indicesS=countAllBoutsSplit(groups,history_length,omitForwards=omitForwards,removeClippedBouts=removeClippedBouts)
    train_set_t = np.zeros((num_train, history_length, num_param))
    test_set_t = np.zeros((num_test, history_length, num_param))
    if prediction_target>1:
        train_goal_t = np.zeros([num_train,prediction_target,num_param])
        test_goal_t = np.zeros([num_test,prediction_target,num_param])
    else:
        train_goal_t = np.zeros([num_train,num_param])
        test_goal_t = np.zeros([num_test,num_param])
    
    if keepGroupsSeperate: train_setS,train_goalS,test_setS,test_goalS=[],[],[],[]

    fish_counter,train_counter,test_counter=0,0,0
    
    ######## START OF GROUP LOOP ########    
    for groupIND,group in enumerate(groups):
        if keepGroupsSeperate:fish_counter,train_counter,test_counter=0,0,0
        ######## START OF FISH LOOP ########
        for boutpath in group:
            train_indices=train_indicesS[fish_counter]
            test_indices=test_indicesS[fish_counter]
            # Load bouts (starts, peaks, stops, durations, dAngle, dSpace, x, y) 
            bouts=np.load(boutpath)['bouts']
            dAngle=bouts[1:,4]
            IBI=np.diff(bouts[:,0]) / FPS 
            dSpace=bouts[1:,5]
            if omitForwards:
                turnsOnly=np.abs(dAngle)>turnThresh
                IBI=IBI[turnsOnly]
                dSpace=dSpace[turnsOnly]
                dAngle=dAngle[turnsOnly]
                
            ######## START OF TRAIN LOOP ########
            for i in train_indices:
                history_range = np.arange(i, i+history_length)
                prediction_range = np.arange(i+history_length, i+history_length+(prediction_target))
                
                train_set_t[train_counter,:,0] = dSpace[history_range]
                train_set_t[train_counter,:,1] = dAngle[history_range]
                train_set_t[train_counter,:,2] = IBI[history_range]
                if prediction_target>1:
                    train_goal_t[train_counter,:,0] = dSpace[prediction_range]
                    train_goal_t[train_counter,:,1] = dAngle[prediction_range]
                    train_goal_t[train_counter,:,2] = IBI[prediction_range]
                else:
                    train_goal_t[train_counter,0] = dSpace[prediction_range]
                    train_goal_t[train_counter,1] = dAngle[prediction_range]
                    train_goal_t[train_counter,2] = IBI[prediction_range]
                train_counter+=1
            ######## END OF TRAIN LOOP ########
            
            ######## START OF TEST LOOP ########
            for ind,i in enumerate(test_indices):
                history_range = np.arange(i - history_length, i)
                prediction_range = np.arange(i, i + prediction_target)
                
                test_set_t[test_counter,:,0] = dSpace[history_range]
                test_set_t[test_counter,:,1] = dAngle[history_range]
                test_set_t[test_counter,:,2] = IBI[history_range]
                if prediction_target>1:
                    test_goal_t[test_counter,:,0] = dSpace[prediction_range]
                    test_goal_t[test_counter,:,1] = dAngle[prediction_range]
                    test_goal_t[test_counter,:,2] = IBI[prediction_range]
                else:
                    test_goal_t[test_counter,0] = dSpace[prediction_range]
                    test_goal_t[test_counter,1] = dAngle[prediction_range]
                    test_goal_t[test_counter,2] = IBI[prediction_range]
                test_counter+=1
            ######## END OF TEST LOOP ########
            if keepGroupsSeperate:
                if shuffle:
                    num_bouts=len(test_set_t)
                    test_set_t = test_set_t[np.random.permutation(np.arange(num_bouts))]
                    num_bouts=len(train_set_t)
                    train_set_t = train_set_t[np.random.permutation(np.arange(num_bouts))]
                    num_bouts=len(test_goal_t)
                    test_goal_t = test_goal_t[np.random.permutation(np.arange(num_bouts))]
                    num_bouts=len(train_goal_t)
                    train_goal_t = train_goal_t[np.random.permutation(np.arange(num_bouts))]
                    
                train_setS.append(train_set_t)
                train_goalS.append(train_goal_t)
                test_setS.append(test_set_t)
                test_goalS.append(test_goal_t)
            
            else:
                if groupIND==len(groups):
                    if shuffle: 
                        num_bouts=len(test_set_t)
                        test_set_t = test_set_t[np.random.permutation(np.arange(num_bouts))]
                        num_bouts=len(train_set_t)
                        train_set_t = train_set_t[np.random.permutation(np.arange(num_bouts))]
                        num_bouts=len(test_goal_t)
                        test_goal_t = test_goal_t[np.random.permutation(np.arange(num_bouts))]
                        num_bouts=len(train_goal_t)
                        train_goal_t = train_goal_t[np.random.permutation(np.arange(num_bouts))]
            
                train_setS=train_set_t
                train_goalS=train_goal_t
                test_setS=test_set_t
                test_goalS=test_goal_t
            ######## END OF FISH LOOP ########   
            fish_counter+=1
       ######## END OF GROUP LOOP ########    
       
    return np.array(train_setS),np.array(train_goalS),np.array(test_setS),np.array(test_goalS)


# Construct test and training sets for the groups
def buildTrainTestSets(groups,omitForwards=False,keepGroupsSeperate=False,shuffle=False,FPS=120,history_length=5,prediction_target=1,train_examples_per_fish=1000,test_examples_per_fish=200):
    
    if len(groups)>4:groups=[groups] # if more than 4, then is unlikely to be a list of lists
    num_fish=0
    if keepGroupsSeperate==False:
        for i in groups:
            num_fish+=len(i)
        
        # Create train / test datasets
        num_train = train_examples_per_fish * num_fish
        num_test = test_examples_per_fish * num_fish

        train_set_t = np.zeros((num_train, history_length, 3))
        test_set_t = np.zeros((num_test, history_length, 3))
        if prediction_target>1:
            train_goal_t = np.zeros([num_train,prediction_target,3])
            test_goal_t = np.zeros([num_test,prediction_target,3])
        else:
            train_goal_t = np.zeros([num_train,3])
            test_goal_t = np.zeros([num_test,3])
        
    train_setS,train_goalS,test_setS,test_goalS=[],[],[],[]

    train_counter,test_counter=0,0    
    for group in groups:
        if keepGroupsSeperate:
            num_fish=len(group)
            # Create train / test datasets
            num_train = train_examples_per_fish * num_fish
            num_test = test_examples_per_fish * num_fish
    
            train_set_t = np.zeros((num_train, history_length, 3))
            test_set_t = np.zeros((num_test, history_length, 3))
            
            if prediction_target>1:
                train_goal_t = np.zeros([num_train,prediction_target,3])
                test_goal_t = np.zeros([num_test,prediction_target,3])
            else:
                train_goal_t = np.zeros([num_train,3])
                test_goal_t = np.zeros([num_test,3])
            
            train_counter,test_counter=0,0
        for boutpath in group:
            # Load bouts (starts, peaks, stops, durations, dAngle, dSpace, x, y) 
            bouts=np.load(boutpath)['bouts']
            dAngle=bouts[0:-1,4]
            dSpace=bouts[0:-1,5]
            IBI=np.diff(bouts[:,0]) / FPS
            num_bouts=len(IBI)
            
            train_indices = np.random.randint(history_length, num_bouts-prediction_target, train_examples_per_fish)
            for i in train_indices:
                history_range = np.arange(i - history_length, i)
                prediction_range = np.arange(i, i + prediction_target)
                
                train_set_t[train_counter,:,0] = dSpace[history_range]
                train_set_t[train_counter,:,1] = dAngle[history_range]
                train_set_t[train_counter,:,2] = IBI[history_range]
                if prediction_target>1:
                    train_goal_t[train_counter,:,0] = dSpace[prediction_range]
                    train_goal_t[train_counter,:,1] = dAngle[prediction_range]
                    train_goal_t[train_counter,:,2] = IBI[prediction_range]
                else:
                    train_goal_t[train_counter,0] = dSpace[prediction_range]
                    train_goal_t[train_counter,1] = dAngle[prediction_range]
                    train_goal_t[train_counter,2] = IBI[prediction_range]
                train_counter+=1
                
            test_indices = np.random.randint(history_length, num_bouts-prediction_target, test_examples_per_fish)
            for ind,i in enumerate(test_indices):
                history_range = np.arange(i - history_length, i)
                prediction_range = np.arange(i, i + prediction_target)
                
                test_set_t[test_counter,:,0] = dSpace[history_range]
                test_set_t[test_counter,:,1] = dAngle[history_range]
                test_set_t[test_counter,:,2] = IBI[history_range]
                if prediction_target>1:
                    test_goal_t[test_counter,:,0] = dSpace[prediction_range]
                    test_goal_t[test_counter,:,1] = dAngle[prediction_range]
                    test_goal_t[test_counter,:,2] = IBI[prediction_range]
                else:
                    test_goal_t[test_counter,0] = dSpace[prediction_range]
                    test_goal_t[test_counter,1] = dAngle[prediction_range]
                    test_goal_t[test_counter,2] = IBI[prediction_range]
                test_counter+=1
        
        if keepGroupsSeperate:
            if shuffle:
                num_bouts=len(test_set_t)
                test_set_t = test_set_t[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(train_set_t)
                train_set_t = train_set_t[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(test_goal_t)
                test_goal_t = test_goal_t[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(train_goal_t)
                train_goal_t = train_goal_t[np.random.permutation(np.arange(num_bouts))]
                
            train_setS.append(train_set_t)
            train_goalS.append(train_goal_t)
            test_setS.append(test_set_t)
            test_goalS.append(test_goal_t)
            
    if keepGroupsSeperate==False:
        if shuffle:
            num_bouts=len(test_set_t)
            test_set_t = test_set_t[np.random.permutation(np.arange(num_bouts))]
            num_bouts=len(train_set_t)
            train_set_t = train_set_t[np.random.permutation(np.arange(num_bouts))]
            num_bouts=len(test_goal_t)
            test_goal_t = test_goal_t[np.random.permutation(np.arange(num_bouts))]
            num_bouts=len(train_goal_t)
            train_goal_t = train_goal_t[np.random.permutation(np.arange(num_bouts))]
            
        train_setS=train_set_t
        train_goalS=train_goal_t
        test_setS=test_set_t
        test_goalS=test_goal_t
                
    return train_setS,train_goalS,test_setS,test_goalS