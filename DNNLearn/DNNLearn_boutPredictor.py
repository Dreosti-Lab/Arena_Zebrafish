# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:04:08 2021

@author: thoma
"""
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------

import sys
sys.path.append(lib_path)
import pandas as pd
import numpy as np
import AZ_utilities as AZU
import AZ_analysis as AZA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.compose import ColumnTransformer
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import seaborn as sns

FPS=120
# load single dictionary and tracking (e.g.)
IndDic_EC=np.load('D:/Movies/DataForAdam/IndDictionaries/EC_B2/210304_EmxGFP_Ctrl_B2_0_ANALYSIS.npy',allow_pickle=True).item()
#fx_EC,fy_EC,_,_,_,_,_,ort_EC,_=AZU.grabTrackingFromFile('D:/Movies/DataForAdam/DataForAdam/ctrlB2/120Hz/Tracking/210304_EmxGFP_Ctrl_B2_0_tracking.npz',sf=0,ef=-1)

IndDic_EA=np.load('D:/Movies/DataForAdam/IndDictionaries/EA_B2/210309_EmxGFP_Asp_B2_1_ANALYSIS.npy',allow_pickle=True).item()
#fx_EA,fy_EA,_,_,_,_,_,ort_EA,_=AZU.grabTrackingFromFile('D:/Movies/DataForAdam/DataForAdam/aspB2/120Hz/Tracking/210309_EmxGFP_Asp_B2_1_tracking.npz',sf=0,ef=-1)
## Convert metrics we want for analysis into a pandas dataframe
EC = pd.DataFrame (IndDic_EC['data'], columns = ['boutDists','boutAngles']).values[0:3000]
EA = pd.DataFrame (IndDic_EA['data'], columns = ['boutDists','boutAngles']).values[0:3000]

# take sets of n (6) rows as the evidence (5) and test(1)
nEvid=5 # number of bouts to use for evidence 
nPred=1 # number of bouts to predict


boutDists0=[]
boutDists1=[]
boutDists2=[]
boutDists3=[]
boutDists4=[]

boutAngle0=[]
boutAngle1=[]
boutAngle2=[]
boutAngle3=[]
boutAngle4=[]

boutDistsPred=[]
boutAnglePred=[]

for j,i in enumerate(EC[0:-6]):
    
    boutDists0.append(i[0])
    boutAngle0.append(i[1])
    i=EC[j+1]
    boutDists1.append(i[0])
    boutAngle1.append(i[1])
    i=EC[j+2]
    boutDists2.append(i[0])
    boutAngle2.append(i[1])
    i=EC[j+3]
    boutDists3.append(i[0])
    boutAngle3.append(i[1])
    i=EC[j+4]
    boutDists4.append(i[0])
    boutAngle4.append(i[1])
    i=EC[j+5]
    boutDistsPred.append(i[0])
    boutAnglePred.append(i[1])
    
data_EC = {'boutDist0':  boutDists0,
           'boutDist1': boutDists1,
           'boutDist2': boutDists2,
           'boutDist3': boutDists3,
           'boutDist4': boutDists4,
           'boutDistPred': boutDistsPred,
           'boutAngle0':  boutAngle0,
           'boutAngle1': boutAngle1,
           'boutAngle2': boutAngle2,
           'boutAngle3': boutAngle3,
           'boutAngle4': boutAngle4,
           'boutAnglePred': boutAnglePred}

df_EC = pd.DataFrame (data_EC)

train_EC_set,test_EC_set=ttsplit(df_EC,test_size=0.2,random_state=42)        

train_set=train_EC_set.drop("boutDistPred",axis=1)
train_set=train_set.drop("boutAnglePred",axis=1)

test_set=test_EC_set.drop("boutDistPred",axis=1)
test_set=test_set.drop("boutAnglePred",axis=1)

train_set_labels=train_EC_set["boutAnglePred"].values#,"boutDistPred"]
test_set_labels=test_EC_set["boutAnglePred"].values#,"boutDistPred"]

train_set_labelsD=train_EC_set["boutDistPred"].values#,"boutDistPred"]
test_set_labelsD=test_EC_set["boutDistPred"].values#,"boutDistPred"]

pipe=Pipeline([('std_scaler',StandardScaler()),])
#prepped_train_EC=pipe.fit_transform(train_EC)
#prepped_test_EC=pipe.fit_transform(test_EC)

attr=list(train_set)
fullPipe=ColumnTransformer([("All",pipe,attr)])

train_prepped=fullPipe.fit_transform(train_set)
test_prepped=fullPipe.fit_transform(test_set)



def build_and_compile_model():
    model = tf.keras.Sequential([
            layers.Dense(20, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(1)
            ])

    model.compile(loss='mse',
                  optimizer='sgd',
                  metrics=["mae"])
    return model

model_ECD=build_and_compile_model()
model_EC=build_and_compile_model()
historyD=model_ECD.fit(train_prepped, train_set_labelsD, epochs=100,validation_data=(test_prepped,test_set_labelsD))
history=model_EC.fit(train_prepped, train_set_labels, epochs=100,validation_data=(test_prepped,test_set_labels))
    
#probability_model = tf.keras.Sequential([model_EC,tf.keras.layers.Softmax()])
#predictions = model_EC.predict(test_prepped)

#predictions


    # 4) Evaluate the model
#testLoss, testAcc = model_EC.evaluate(test_prepped, test_set_labels, verbose=2)
    
    
    
#train_EC,test_EC=ttsplit(df_EC,test_size=0.2,random_state=42)
#train_EA,test_EA=ttsplit(df_EC,test_size=0.2,random_state=42)


# Standardise (Zero Mean Unit Variance)
#xC = df_EC.loc[:, ['boutDists','boutAngles']].values
#xA = df_EC.loc[:, ['boutDists','boutAngles']].values


