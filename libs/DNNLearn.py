# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:20:50 2021

@author: thoma
"""
import numpy as np
import tensorflow as tf
import random



# Function to test if the model can learn which fish are ablated from inputs
def trainSupervised1DImageClassifier(trainImages,trainLabels,testImages=[],testLabels=[],testProp=0.2,epochs=10):
    
    if testImages==[]: # if no test set provided, take a random subset of testProp=20% of the trainingSet to test the model
        while np.floordivide(len(testLabels),len(trainLabels))<testProp:
            r=np.int(random.randrange(0,len(trainImages)))
            testImages.append(trainImages.pop(r))
            testLabels.append(trainLabels.pop(r))
            
         
    # Normalise test and train datasets identically
    
    if np.max(trainImages)>=np.max(testImages):
        norm=np.max(trainImages)
    else:
        norm=np.max(testImages)
        
    trainImages, testImages = np.divide(trainImages,norm), np.divide(testImages,norm)
    
    numClass=len(np.unique(trainLabels))
    
    # 2a) Build layers of neural net
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(numClass)
            ])

    
    
    # 2b) Compile with defined loss function, optimiser and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
#    # Check initial values for loss and prediction accuracy
#    InitialPredictions = model(trainImages[:1]).numpy()
#    tf.nn.softmax(InitialPredictions).numpy()
#    loss_fn(y_train[:1], InitialPredictions).numpy()
    
    # 3) Train the model
    model.fit(trainImages, trainLabels, epochs=epochs)
    
    # 4) Evaluate the model
    testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)
    
    return model,testLoss,testAcc

# Function to test if the model can learn which fish are ablated from inputs
def trainSupervised2DImageClassifier(trainImages,trainLabels,testImages=[],testLabels=[],testProp=0.2,epochs=10):
    
    if testImages==[]: # if no test set provided, take a random subset of testProp=20% of the trainingSet to test the model
        while np.floordivide(len(testLabels),len(trainLabels))<testProp:
            r=np.int(random.randrange(0,len(trainImages)))
            testImages.append(trainImages.pop(r))
            testLabels.append(trainLabels.pop(r))
            
         
    # Normalise test and train datasets identically
    trainImages
    if np.max(trainImages)>=np.max(testImages):
        norm=np.max(trainImages)
    else:
        norm=np.max(testImages)
        
    trainImages, testImages = np.divide(trainImages,norm), np.divide(testImages,norm)
    
    imHeight=len(trainImages[0,0,:])
    imWidth=len(trainImages[0,:,0])
    numClass=len(np.unique(trainLabels))
    
    # 2a) Build layers of neural net
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(imWidth, imHeight)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(numClass)
            ])

    
    
    # 2b) Compile with defined loss function, optimiser and metrics
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
#    # Check initial values for loss and prediction accuracy
#    InitialPredictions = model(trainImages[:1]).numpy()
#    tf.nn.softmax(InitialPredictions).numpy()
#    loss_fn(y_train[:1], InitialPredictions).numpy()
    
    # 3) Train the model
    model.fit(trainImages, trainLabels, epochs=epochs)
    
    # 4) Evaluate the model
    testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)
    
    return model,testLoss,testAcc

def grabImagesAndLabels(Dic,label=0,trainImages=[],trainLabels=[],smallest=1000000,metric='distPerFrame'):
    
    flag=0
    if len(trainImages)!=0:
        smallest=len(trainImages[0])
        
    for thisFish in Dic['Ind_fish']:
        s=len(thisFish['data'][metric])
        
        if s < smallest : 
            smallest = s
            flag=1
            
    if len(trainImages)!=0 and flag==1:
        newTI=[]
        for prevFish in trainImages:
            newTI.append(prevFish[0:smallest])
        trainImages=newTI
        
    for thisFish in Dic['Ind_fish']:
        thisDistPerFrame=thisFish['data'][metric]
        thisFishData=thisDistPerFrame.tolist()
        thisFishData=thisFishData[0:smallest]
        trainImages.append(thisFishData)
        trainLabels.append(label)
        
#    trainImages,trainLabels= np.array(trainImages),np.array(trainLabels)
    return trainImages,trainLabels

    
    