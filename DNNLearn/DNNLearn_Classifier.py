# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:35:33 2021

@author: thoma
"""
lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
#-----------------------------------------------------------------------------

import sys
sys.path.append(lib_path)
import numpy as np
import tensorflow as tf
import DNNLearn as DNN

# take x for tracking and y for tracking
# cut up into 10 second chunks
# 80% is training, 20% is testing (from 700 sets, 560 sequences to training per fish, 140 to testing) 
# cut each sequence in half; first half trainer, second half label
# concatenate training and testing sets for each fish
# create 2D array of all fish training and test sets concatenated


#mnist = tf.keras.datasets.mnist
#(trainImages,trainLabels), (testImages, testLabels) = mnist.load_data()
# %% Load data and create training and testing sets
Dic_EC_B0=np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EC_B0.npy',allow_pickle=True).item()
Dic_EA_B0=np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EA_B0.npy',allow_pickle=True).item()

Dic_EC_M0=np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EC_M0.npy',allow_pickle=True).item()
Dic_EA_M0=np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EA_M0.npy',allow_pickle=True).item()

trainImages_B0,trainLabels_B0 = DNN.grabImagesAndLabels(Dic_EC_B0,label=0,trainImages=[],trainLabels=[],metric='dispersal')
trainImages_B0,trainLabels_B0 = DNN.grabImagesAndLabels(Dic_EA_B0,label=1,trainImages=trainImages_B0,trainLabels=trainLabels_B0,metric='dispersal')

trainImages_M0,trainLabels_M0 = DNN.grabImagesAndLabels(Dic_EC_M0,label=0,trainImages=[],trainLabels=[],metric='dispersal')
trainImages_M0,trainLabels_M0 = DNN.grabImagesAndLabels(Dic_EA_M0,label=1,trainImages=trainImages_M0,trainLabels=trainLabels_M0,metric='dispersal')

ind=[0,1,1,3,1,-1,-2,-2,-1]

def subsetSample(im,lab,ind):
    subIm=[]
    subLab=[]
    for i in ind:
        subIm.append(im.pop(i))
        subLab.append(lab.pop(i))
        
    return im,lab,subIm,subLab 

trainImages_M0,trainLabels_M0,testImages_M0,testLabels_M0=subsetSample(trainImages_M0,trainLabels_M0,ind)
trainImages_B0,trainLabels_B0,testImages_B0,testLabels_B0=subsetSample(trainImages_B0,trainLabels_B0,ind)
# %%
trainImages_B0=np.array(trainImages_B0)
trainImages_M0=np.array(trainImages_M0)
trainLabels_B0=np.array(trainLabels_B0)
trainLabels_M0=np.array(trainLabels_M0)

testImages_B0=np.array(testImages_B0)
testImages_M0=np.array(testImages_M0)
testLabels_B0=np.array(testLabels_B0)
testLabels_M0=np.array(testLabels_M0)
    
# %% Train model and generate predictions on testSet
model,loss,acc=DNN.trainSupervised1DImageClassifier(trainImages_B0,trainLabels_B0,testImages=testImages_B0,testLabels=testLabels_B0)

# create softmax layer in the model to convert "logits" output to probabilities
# WE DO NOT DO THIS IN THE MODEL WHEN TRAINING AS IT INRTODUCES AMBIGUITY
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

# explicitly make predictions from (new) images
predictions = probability_model.predict(testImages_B0)
# this outputs a list of confidence values per image between 0 and 1 for each category. The category with the highest confidence should be picked
predClasses_B0=[]
for pred in predictions:
    predClasses_B0.append(np.argmax(pred))

