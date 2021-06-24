# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:24:54 2021

@author: thoma
"""
# %% LOAD DATA, OPTIONS AND PREPARE VARIABLES
DATAROOT = r'D:\Movies/DataForAdam/DataForAdam/'
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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

import ARK_utilities
import ARK_bouts
# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# %% load data
# Get bout files (controls)
bout_paths_controls = []
bout_paths_controls += glob.glob(DATAROOT + r"/GroupedTracking/EC_M0/*bouts.npz")
bout_paths_controls_B0 = []
bout_paths_controls_B0 += glob.glob(DATAROOT + "/GroupedTracking/EC_B0/*bouts.npz")

# Get bout files (ablation)
bout_paths_lesions = []
bout_paths_lesions += glob.glob(DATAROOT + r"/GroupedTracking/EA_M0/*bouts.npz")
bout_paths_lesions_B0 = []
bout_paths_lesions_B0 += glob.glob(DATAROOT + "/GroupedTracking/EA_B0/*bouts.npz")

# Parameters
FPS=120
history_length = 5 # 
prediction_target = 1 # 
train_examples_per_fish = 2000
test_examples_per_fish = 500

# Set groups and empty results arrays
groups = [bout_paths_controls, bout_paths_lesions] # [bout_paths_controls_B0,bout_paths_lesions_B0]
train_set_control=[]   
train_goal_control=[]

train_set_lesion=[]
train_goal_lesion=[]

test_set_control=[]
test_goal_control=[]

test_set_lesion=[]
test_goal_lesion=[]

# %% Construct test and training sets for two groups
def buildTrainTestSets(groups,shuffle=False,FPS=120,train_examples_per_fish=1000):
    
    if len(groups)>2:groups=[groups] # if more than 2, then is unlikely to be a list of lists
    num_fish=0
    for i in groups:
        num_fish+=len(i)
        
    # Create train / test datasets
    num_train = train_examples_per_fish * num_fish
    num_test = test_examples_per_fish * num_fish

    train_set = np.zeros((num_train, history_length, 3))
    test_set = np.zeros((num_test, history_length, 3))

    train_goal = np.zeros([num_train,3])
    test_goal = np.zeros([num_test,3])
    
    train_counter=0
    test_counter=0
    for group in groups:
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

train_set_control,train_goal_control,xTest_control,yTest_control = buildTrainTestSets(groups[0],train_examples_per_fish=train_examples_per_fish)
train_set_lesion,train_goal_lesion,xTest_lesion,yTest_lesion = buildTrainTestSets(groups[1],train_examples_per_fish=train_examples_per_fish)
_,_,xTest_control_shuffle,yTest_control_shuffle = buildTrainTestSets(groups[0],shuffle=True,train_examples_per_fish=train_examples_per_fish)
_,_,xTest_lesion_shuffle,yTest_lesion_shuffle = buildTrainTestSets(groups[1],shuffle=True,train_examples_per_fish=train_examples_per_fish)

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

xTrain=train
yTrain=goal #xTrain,xValid,yTrain,yValid = train_test_split(train,goal)

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
flattenList=[xTrain,xTest_lesion,xTest_control] #flattenList=[xTrain,xValid,xTest_lesion,xTest_control] # flatten x lists in preparation for scaling
columnnames=[]
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
flatList=[]
for i in flattenList:
    flatList.append(flattenForModel(i,columnnames))
    
#xTrain,xValid=flatList[0],flatList[1],flatList[2],flatList[3]

# %% Scale data
preppedX=[]
attribs=list(flatList[0])
scaler=RobustScaler()
for thisDataSet in flatList:
    preppedX.append(scaler.fit_transform(thisDataSet))

#scaler=StandardScaler()
#xToTrain = scaler.fit_transform(xTrain)
#yToTrain = scaler.fit_transform(yTrain)
#xToValid = scaler.fit_transform(xValid)
#yToValid = scaler.fit_transform(yValid)
#xToTest_control = scaler.fit_transform(xTest_control)
#yToTest_control = scaler.fit_transform(yTest_control)
#xToTest_lesion = scaler.fit_transform(xTest_lesion)
#yToTest_lesion = scaler.fit_transform(yTest_lesion)

# %% Define and compile model
def build_and_compile_model(nNeurons,loss,opt,met,input_shape=(5,3)):
    
    model = tf.keras.models.Sequential()
    for thisLayer in nNeurons[0:-1]:
        model.add(tf.keras.layers.Dense(thisLayer, activation='relu'))
    model.add(tf.keras.layers.Dense(nNeurons[-1]))
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[met])
    return model

# Create models to predict distance, angle and IBI, seperately, then together
#nNeurons=[20,10,1]
#
#model_Dist=build_and_compile_model(nNeurons,'mse','adam','mae')
#model_Angle=build_and_compile_model(nNeurons,'mse','adam','mae')
#model_IBI=build_and_compile_model(nNeurons,'mse','adam','mae')
epochs=500
nNeurons=[15,300,300,3]
model_Combined=build_and_compile_model(nNeurons,'mse','adam','mae')

# %% Train models
train=preppedX[0]
train_valid=preppedX[1]
test_lesion=preppedX[2]
test_control=preppedX[3]
goal=yTrain
#goal_valid=yValid
goal_test_control=yTest_control
goal_test_lesion=yTest_lesion

#history_Dist=model_Dist.fit(train[:,12], goal[:,0], epochs=epochs)
#history_Angle=model_Angle.fit(train[:,13], goal[:,1], epochs=epochs)
#history_IBI=model_IBI.fit(train[:,14], goal[:,2], epochs=epochs)

history_Combined=model_Combined.fit(train, goal, epochs=epochs)

# %% Performance
## Combined model
label='Combined - Dist, Angle, IBI'
# make predictions on test sets
predControl=model_Combined.predict(test_control)
predLesion=model_Combined.predict(test_lesion)

# compare to true
mae_ctrl_true = np.mean(np.abs(predControl - yTest_control),axis=0)
mae_lesion_true = np.mean(np.abs(predLesion - yTest_lesion),axis=0)

# compare to shuffled
mae_ctrl_shuffle = np.mean(np.abs(predControl - yTest_control_shuffle),axis=0)
mae_lesion_shuffle = np.mean(np.abs(predLesion - yTest_lesion_shuffle),axis=0)

# relative maes
print('Relative MAE Control = ' + label + ' ' + str(mae_ctrl_true / mae_ctrl_shuffle))
print('Relative MAE Lesion = ' + label + ' ' + str(mae_lesion_true / mae_lesion_shuffle))

# %% Individual models

## Dist model
#label='Displacement'
## make predictions on test sets
#predControl=model_Dist.predict(test_control)
#predLesion=model_Dist.predict(test_lesion)
#
## compare to true
#mae_ctrl_true = np.mean(np.abs(predControl - yTest_control[:,0]),axis=0)
#mae_lesion_true = np.mean(np.abs(predLesion - yTest_lesion[:,0]),axis=0)
#
## compare to shuffled
#mae_ctrl_shuffle = np.mean(np.abs(predControl - yTest_control_shuffle[:,0]),axis=0)
#mae_lesion_shuffle = np.mean(np.abs(predLesion - yTest_lesion_shuffle[:,0]),axis=0)
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
#mae_ctrl_shuffle = np.mean(np.abs(predControl - yTest_control_shuffle[:,0]),axis=0)
#mae_lesion_shuffle = np.mean(np.abs(predLesion - yTest_lesion_shuffle[:,0]),axis=0)
#
## relative maes
#print('Relative MAE Control = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_ctrl_true / mae_ctrl_shuffle))))
#print('Relative MAE Lesion = ' + ' ' + label + ' ' + str(np.mean(np.abs(mae_lesion_true / mae_lesion_shuffle))))
#
### IBI model
#label='IBI'
## make predictions on test sets
#predControl=model_Angle.predict(test_control)
#predLesion=model_Angle.predict(test_lesion)
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

