# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:24:54 2021

@author: thoma
"""
# %% LOAD DATA, OPTIONS AND PREPARE VARIABLES
DATAROOT = r'D:\Movies\DataForAdam\DataForAdam\GroupDictionaries'
LIBROOT = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish'

# Set library paths
import sys
lib_path = LIBROOT + "/ARK/libs"
ARK_lib_path = LIBROOT + "/libs"
sys.path.append(lib_path)
sys.path.append(ARK_lib_path)

# Import useful libraries
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

import ARK_utilities
import ARK_bouts
# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# %% load data
# Get group dict files (controls)
#dict_paths_controls = []
dict_path_controls = glob.glob(DATAROOT + r"/EC_M0.npy")


# Get bout files (ablation)
dict_path_lesions = glob.glob(DATAROOT + r"/EA_M0.npy")

# Parameters
FPS=120
history_length = 5 # 
prediction_target = 1 # 
train_examples_per_fish = 2000
test_examples_per_fish = 500

# Set groups and empty results arrays
groups = [dict_path_controls, dict_path_lesions] # [bout_paths_controls_B0,bout_paths_lesions_B0]
train_set_control=[]   
train_goal_control=[]

train_set_lesion=[]
train_goal_lesion=[]

test_set_control=[]
test_goal_control=[]

test_set_lesion=[]
test_goal_lesion=[]

# %% Construct test and training sets for two groups
def buildTrainTestSets(groups,shuffle=False,FPS=120,train_examples_per_fish=1000,test_examples_per_fish=200):
    
    num_fish=0
    for i in groups:
        a=np.load(i,allow_pickle=True).item()
        a=a['Ind_fish']
        num_fish+=len(a)
        
    # Create train / test datasets
    num_train = train_examples_per_fish * num_fish
    num_test = test_examples_per_fish * num_fish

    train_set = np.zeros((num_train, history_length, 3))
    test_set = np.zeros((num_test, history_length, 3))

    train_goal = np.zeros([num_train,3])
    test_goal = np.zeros([num_test,3])
    
    train_counter=0
    test_counter=0
    for dicpath in groups:
        dic=np.load(dicpath,allow_pickle=True).item()
        for thisFish in dic['Ind_fish']:
            dAngle=thisFish['data']['boutAngles'][0:-1]
            dSpace=thisFish['data']['boutDists'][0:-1]
            IBI=np.diff(np.divide(thisFish['data']['boutStarts'], FPS))
            num_bouts=len(IBI)
        
            train_indices = np.random.randint(history_length, num_bouts-prediction_target, train_examples_per_fish)
#            print(len(train_indices))
            for i in train_indices:
                if history_length==0:
                    history_range = i
                else:
                    history_range = np.arange(i - history_length, i)
        
                train_set[train_counter,:,0] = dSpace[history_range]
                train_set[train_counter,:,1] = dAngle[history_range]
                train_set[train_counter,:,2] = IBI[history_range]
                train_goal[train_counter,0] = dSpace[i + prediction_target]
                train_goal[train_counter,1] = dAngle[i + prediction_target]
                train_goal[train_counter,2] = IBI[i + prediction_target]
                train_counter+=1
                    
            test_indices = np.random.randint(history_length, num_bouts-prediction_target, test_examples_per_fish)
            for ind,i in enumerate(test_indices):
                history_range = np.arange(i - history_length, i)
                
                test_set[test_counter,:,0] = dSpace[history_range]
                test_set[test_counter,:,1] = dAngle[history_range]
                test_set[test_counter,:,2] = IBI[history_range]
                test_goal[test_counter,0] = dSpace[i + prediction_target]
                test_goal[test_counter,1] = dAngle[i + prediction_target]
                test_goal[test_counter,2] = IBI[i + prediction_target]
                test_counter+=1
            if shuffle:
                num_bouts=len(test_set)
                test_set = test_set[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(train_set)
                train_set = train_set[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(test_goal)
                test_goal = test_goal[np.random.permutation(np.arange(num_bouts))]
                num_bouts=len(train_goal)
                train_goal = train_goal[np.random.permutation(np.arange(num_bouts))]
        
    return train_set,train_goal,test_set,test_goal

train_set_control,train_goal_control,xTest_control,yTest_control = buildTrainTestSets(groups[0])
train_set_lesion,train_goal_lesion,xTest_lesion,yTest_lesion = buildTrainTestSets(groups[1])
_,_,xTest_control_shuffle,yTest_control_shuffle = buildTrainTestSets(groups[0],shuffle=True)
_,_,xTest_lesion_shuffle,yTest_lesion_shuffle = buildTrainTestSets(groups[1],shuffle=True)

# %% finalise training and set sets
trainlist=train_set_control.tolist()
trainlistc=train_set_lesion.tolist()
for i in (trainlistc):
    trainlist.append(i)
    
train_goal=train_goal_control.tolist()
train_goalc=train_goal_lesion.tolist()
for i in (train_goalc):
    train_goal.append(i)

train=np.array(trainlist)
goal=np.array(train_goal)

xTrain,xValid,yTrain,yValid = train_test_split(train,goal)
# flatten internal lists
def flattenForModelDF(vec,colnames):
    for i,thisBoutSet in enumerate(vec):
        for j,thisBout in enumerate(thisBoutSet):
            for thisMet in thisBout:
                if j==0:
                    data = {colnames[j]:  thisMet}
                else: 
                    data[colnames[j]]=thisMet
            dftoAdd = pd.DataFrame (data, columns = [colnames])
        if i ==0:
            df=dftoAdd
        else:
            df.append(dftoAdd)
    return df
# flatten internal lists
def flattenForModel(vec,colnames):
    
    vec_flat=[]
    for i,thisBoutSet in enumerate(vec):
#        print(i/len(vec))
        vec_flat_temp=[]
        for j,thisBout in enumerate(thisBoutSet):
            for thisMet in thisBout:
                vec_flat_temp.append(thisMet)
        
        vec_flat.append(vec_flat_temp)
        
    df = pd.DataFrame (vec_flat, columns = colnames)
    return df
# flatten internal lists
def flattenYForModel(vec,colnames):
    
    vec_flat=[]
    for i,thisBout in enumerate(vec):
#        print(i/len(vec))
        vec_flat_temp=[]
        for thisMet in thisBout:
            vec_flat_temp.append(thisMet)
        
        vec_flat.append(vec_flat_temp)
        
    df = pd.DataFrame (vec_flat, columns = colnames)
    return df
flattenList=[xTrain,xValid,xTest_lesion,xTest_control,xTest_lesion_shuffle,xTest_control_shuffle] # flatten x lists in preparation for scaling
flattenY=[yTest_control,yTest_lesion,yTest_control_shuffle,yTest_lesion_shuffle,yTrain,yValid]
columnnames=[]
columnYnames=[]
#distAttrib=[]
#angAttrib=[]
#IBIAttrib=[]
for j in range(len(xTrain[0])):
    columnnames.append('BoutDist'+str(j))
#    distAttrib.append('BoutDist'+str(j))
    columnnames.append('BoutAngle'+str(j))
#    angAttrib.append('BoutAngle'+str(j))
    columnnames.append('BoutIBI'+str(j))
#    IBIAttrib.append('BoutIBI'+str(j))
columnYnames.append('BoutDist')
columnYnames.append('BoutAngle')
columnYnames.append('BoutIBI')
flatList=[]
flatYList=[]
for i in flattenList:
    flatList.append(flattenForModel(i,columnnames))
for i in flattenY:
    flatYList.append(flattenYForModel(i,columnYnames))
    
    
#xTrain,xValid=flatList[0],flatList[1],flatList[2],flatList[3]

# %% Scale data
scale=True
preppedX=[]
preppedY=[]

if scale:

    logAttr=[]
    stAttr=[]
    logAttrY=[]
    logAttrY.append('BoutDist')
    logAttrY.append('BoutIBI')
    if history_length==1:
        logAttr.append('BoutDist0')
        logAttr.append('BoutIBI0')
        stAttr.append('BoutAngle0')
    else:
        for i in range(0,history_length):
            logAttr.append('BoutDist' + str(i))
            logAttr.append('BoutIBI' + str(i))
            stAttr.append('BoutAngle' + str(i))
        
    scaler=StandardScaler()
    for i,thisDataSet in enumerate(flatList):
        for logThis in logAttr:
            thisDataSet[logThis]=np.log(thisDataSet[logThis]+1)
        if i==0:
            preppedX.append(scaler.fit_transform(thisDataSet))
        else: 
            preppedX.append(scaler.transform(thisDataSet))
    
    for i,thisDataSet in enumerate(flatYList):
        for logThis in logAttrY:
            thisDataSet[logThis]=np.log(thisDataSet[logThis]+1)
        if i==0:
            preppedY.append(scaler.fit_transform(thisDataSet))
        else: 
            preppedY.append(scaler.transform(thisDataSet))
            
else: 
    for k in range(len(flatList)):
        preppedX.append(np.array(flatList[k]))
    for k in range(len(flatYList)):
        preppedY.append(np.array(flatYList[k]))

    # %% Define and compile regression perceptron model
def build_and_compile_model(nNeurons,loss,opt,met,input_shape=(5,3)):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(nNeurons[0],activation='relu',input_shape=input_shape))
    for thisLayer in nNeurons[1:-1]:
        model.add(tf.keras.layers.Dense(thisLayer, activation='relu'))
    model.add(tf.keras.layers.Dense(nNeurons[-1]))
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[met])
    return model

# Create models to predict distance, angle and IBI, seperately by regression, then together in an MLP

train=preppedX[0]
train_valid=preppedX[1]
test_lesion=preppedX[2]
test_control=preppedX[3]
test_lesion_shuffle=preppedX[4]
test_control_shuffle=preppedX[5]

goal_test_control=preppedY[0]
goal_test_lesion=preppedY[1]
goal_test_control_shuffle=preppedY[2]
goal_test_lesion_shuffle=preppedY[3]
goal=preppedY[4]
goal_valid=preppedY[5]

lastBoutTrain_NoAngle=train[:,[12,14]]
lastbouttest_control_NoAngle=test_control[:,[12,14]]
lastbouttest_lesion_NoAngle=test_lesion[:,[12,14]]
lastboutgoal_test_control_NoAngle=goal_test_control[:,[0,2]]
lastboutgoal_test_lesion_NoAngle=goal_test_lesion[:,[0,2]]
lastBoutGoal_NoAngle=goal[:,[0,2]]

nNeurons=[2,20,2]
model_lastBout_NoAngle=build_and_compile_model(nNeurons,'mse','adam','mae',input_shape=lastBoutTrain_NoAngle.shape[1:])

lastBoutTrain=train[:,11:14]
lastbouttest_control=test_control[:,11:14]
lastbouttest_lesion=test_lesion[:,11:14]
lastboutgoal_test_control=goal_test_control
lastboutgoal_test_lesion=goal_test_lesion
lastBoutGoal=goal

nNeurons=[3,300,3]
model_lastBout=build_and_compile_model(nNeurons,'mae','adam','mae',input_shape=lastBoutTrain.shape[1:])
#model_Angle=build_and_compile_model(nNeurons,'mae','sgd','mse')
#model_IBI=build_and_compile_model(nNeurons,'mae','sgd','mse')
nNeurons=[15,300,300,3]
model_5Bouts=build_and_compile_model(nNeurons,'mae','adam','mae',input_shape=train.shape[1:])

# %% Train models
epochs=2000


#weightDist,biasDist=LinRegModel(train[:,0],goal[:,0],epochs=1000,learning_rate=0.01)
#weightAngle,biasAngle=LinRegModel(train[:,1],goal[:,1],epochs=1000,learning_rate=0.01)
#weightIBI,biasIBI=LinRegModel(train[:,2],goal[:,2],epochs=1000,learning_rate=0.01)

history_lastBout_NoAngle=model_lastBout_NoAngle.fit(lastBoutTrain_NoAngle, lastBoutGoal_NoAngle, epochs=epochs,shuffle=True,batch_size=5000)
history_lastBout=model_lastBout.fit(lastBoutTrain, lastBoutGoal, epochs=epochs,shuffle=True,batch_size=5000)
history_5bouts=model_5Bouts.fit(train, goal, epochs=epochs,shuffle=True,batch_size=5000)

# %% Performance
## Last bout model
label='Last Bout'
# make predictions on test sets


predLastBout_control=model_lastBout.predict(lastbouttest_control)
predLastBout_lesion=model_lastBout.predict(lastbouttest_lesion)

mae_lastBoutctrl_true = np.mean(np.abs(predLastBout_control - lastboutgoal_test_control),axis=0)
mae_lastBoutlesion_true = np.mean(np.abs(predLastBout_lesion - lastboutgoal_test_lesion),axis=0)

mae_lastBoutctrl_random = np.mean(np.abs(predLastBout_control - np.random.permutation(lastboutgoal_test_control)),axis=0)
mae_lastBoutlesion_random = np.mean(np.abs(predLastBout_lesion - np.random.permutation(lastboutgoal_test_lesion)),axis=0)

print('Relative MAE Control = ' + label + ' ' + str(mae_lastBoutctrl_true / mae_lastBoutctrl_random))
print('Relative MAE Lesion = ' + label + ' ' + str(mae_lastBoutlesion_true / mae_lastBoutlesion_random))


# %% Last bout no angle model
label='Last Bout no angle'
# make predictions on test sets

predLastBout_control_NoAngle=model_lastBout_NoAngle.predict(lastbouttest_control_NoAngle)
predLastBout_lesion_NoAngle=model_lastBout_NoAngle.predict(lastbouttest_lesion_NoAngle)

mae_lastBoutctrl_true_NoAngle = np.mean(np.abs(predLastBout_control_NoAngle - lastboutgoal_test_control_NoAngle),axis=0)
mae_lastBoutlesion_true_NoAngle = np.mean(np.abs(predLastBout_lesion_NoAngle - lastboutgoal_test_lesion_NoAngle),axis=0)

mae_lastBoutctrl_random_NoAngle = np.mean(np.abs(predLastBout_control_NoAngle - np.random.permutation(lastboutgoal_test_control_NoAngle)),axis=0)
mae_lastBoutlesion_random_NoAngle = np.mean(np.abs(predLastBout_lesion_NoAngle - np.random.permutation(lastboutgoal_test_lesion_NoAngle)),axis=0)

print('Relative MAE Control = ' + label + ' ' + str(mae_lastBoutctrl_true_NoAngle / mae_lastBoutctrl_random_NoAngle))
print('Relative MAE Lesion = ' + label + ' ' + str(mae_lastBoutlesion_true_NoAngle / mae_lastBoutlesion_random_NoAngle))

# %% Combined model
label='Last 5 bouts'

predControl=model_5Bouts.predict(test_control)
predLesion=model_5Bouts.predict(test_lesion)

#predControl_shuffle=model_5Bouts.predict(test_control_shuffle)
#predLesion_shuffle=model_5Bouts.predict(test_lesion_shuffle)

# compare to true
mae_ctrl_true = np.mean(np.abs(predControl - goal_test_control),axis=0)
mae_lesion_true = np.mean(np.abs(predLesion - goal_test_lesion),axis=0)

# compare to shuffled
mae_ctrl_shuffle = np.mean(np.abs(predControl - np.random.permutation(goal_test_control)),axis=0)
mae_lesion_shuffle = np.mean(np.abs(predLesion - np.random.permutation(goal_test_lesion)),axis=0)

# relative maes
print('Relative MAE Control = ' + label + ' ' + str(mae_ctrl_true / mae_ctrl_shuffle))
print('Relative MAE Lesion = ' + label + ' ' + str(mae_lesion_true / mae_lesion_shuffle))

## Dist model
#label='Displacement'
### make predictions on test sets
#predControl=model_Dist.predict(test_control)
#predLesion=model_Dist.predict(test_lesion)
#
#predControl_shuffle=model_Dist.predict(test_control_shuffle)
#predLesion_shuffle=model_Dist.predict(test_lesion_shuffle)
##
### compare to true
##mae_ctrl_true = np.mean(np.abs(predControl - goal_test_control[:,0]),axis=0)
##mae_lesion_true = np.mean(np.abs(predLesion - goal_test_lesion[:,0]),axis=0)
##
### compare to shuffled
#mae_ctrl_shuffle = np.mean(np.abs(predControl - goal_test_control_shuffle[:,0]),axis=0)
#mae_lesion_shuffle = np.mean(np.abs(predLesion - goal_test_lesion_shuffle[:,0]),axis=0)
#
## relative maes
#print('Relative MAE Control = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_ctrl_true / mae_ctrl_shuffle))))
#print('Relative MAE Lesion = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_lesion_true / mae_lesion_shuffle))))
#
### Angle model
#label='Angle'
## make predictions on test sets
#predControl=model_Angle.predict(test_control)
#predLesion=model_Angle.predict(test_lesion)
#
## compare to true
#mae_ctrl_true = np.mean(np.abs(predControl - yTest_control[:,1]),axis=0)
#mae_lesion_true = np.mean(np.abs(predLesion - yTest_lesion[:,1]),axis=0)
#
## compare to shuffled
#mae_ctrl_shuffle = np.mean(np.abs(predControl - yTest_control_shuffle[:,1]),axis=0)
#mae_lesion_shuffle = np.mean(np.abs(predLesion - yTest_lesion_shuffle[:,1]),axis=0)
#
## relative maes
#print('Relative MAE Control = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_ctrl_true / mae_ctrl_shuffle))))
#print('Relative MAE Lesion = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_lesion_true / mae_lesion_shuffle))))
#
### IBI model
#label='IBI'
## make predictions on test sets
#predControl=model_IBI.predict(test_control)
#predLesion=model_IBI.predict(test_lesion)
#
## compare to true
#mae_ctrl_true = np.mean(np.abs(predControl - yTest_control[:,2]),axis=0)
#mae_lesion_true = np.mean(np.abs(predLesion - yTest_lesion[:,2]),axis=0)
#
## compare to shuffled
#mae_ctrl_shuffle = np.mean(np.abs(predControl - yTest_control_shuffle[:,2]),axis=0)
#mae_lesion_shuffle = np.mean(np.abs(predLesion - yTest_lesion_shuffle[:,2]),axis=0)
#
## relative maes
#print('Relative MAE Control = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_ctrl_true / mae_ctrl_shuffle))))
#print('Relative MAE Lesion = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_lesion_true / mae_lesion_shuffle))))

# %% Sanity check on regression models
## manually grab weights and biases from each model
#weights_dist = model_Dist.layers[0].get_weights()[0]
#biases_dist = model_Dist.layers[0].get_weights()[1]
#weights_Angle = model_Angle.layers[0].get_weights()[0]
#biases_Angle = model_Angle.layers[0].get_weights()[1]
#weights_IBI = model_IBI.layers[0].get_weights()[0]
#biases_IBI = model_IBI.layers[0].get_weights()[1]
###
#### scatter plot of Disp, Angle and IBI with regression lines drawn manuall and with predict fnction
#plt.subplot(1,3,1)
#plt.title('Displacement')
#plt.scatter(train[:,0], goal[:,0], alpha=0.1)
#plt.ylabel('Displacement this bout')
#plt.xlabel('Displacement last bout')
#xx=np.linspace(np.min(train[:,0]),np.max(train[:,0]),num=1000)
#yy=(weights_dist[0]*xx)+biases_dist[0]
#plt.plot(xx,yy,color='black',label='Linear Regression')
#yy=model_Dist.predict(xx)
#plt.plot(xx,yy,color='green',label='NN Regression')
#plt.subplot(1,3,2)
#plt.title('Angle')
#plt.scatter(train[:,1], goal[:,1], alpha=0.1)
#plt.ylabel('Angle this bout')
#plt.xlabel('Angle last bout')
#xx=np.linspace(np.min(train[:,1]),np.max(train[:,1]),num=1000)
#yy=(weights_Angle[0]*xx)+biases_Angle[0]
#plt.plot(xx,yy,color='black',label='Linear Regression')
#yy=model_Angle.predict(xx)
#plt.plot(xx,yy,color='green',label='NN Regression')
#plt.subplot(1,3,3)
#plt.title('IBI')
#plt.scatter(train[:,2], goal[:,2], alpha=0.1)
#plt.ylabel('IBI this bout')
#plt.xlabel('IBI last bout')
#xx=np.linspace(np.min(train[:,2]),np.max(train[:,2]),num=1000)
#yy=(weights_IBI[0]*xx)+biases_IBI[0]
#plt.plot(xx,yy,color='black',label='Linear Regression')
#yy=model_IBI.predict(xx)
#plt.plot(xx,yy,color='green',label='NN Regression')
