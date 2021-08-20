# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:33:05 2021

@author: thoma
"""

# Generate multi-variate time series from datasets and prepare for recursive neural network

# LOAD DATA, OPTIONS AND PREPARE VARIABLES
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
import matplotlib.pyplot as plt
import AZ_model as AZMo
import AZ_utilities as AZU
import ARK_utilities
import ARK_bouts
# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)


# Parameters
GroupNameS = ['EC_B0','EA_M0','EC_B0','EA_B0']
scale=True
plot=True
subplot=False
keepFigures=False
saveFig=True
omitForwards=False
LSTM=False

pointsize=5
FPS=120
history_length = 50 # 
prediction_target = 1 # 
numParams = 3
#paramLabels=['boutDist',boutIBI']
paramLabels=['boutDist','boutAngle','boutIBI']
epochs=50
nNeurons=[numParams,48,24,12,numParams*prediction_target]
nNeuronsSingle=[1,48,24,12,1]
#nNeurons=[numParams,36,36,36,numParams*prediction_target]
##nNeurons=[numParams,24,24,24,numParams*prediction_target]
#nNeuronsSingle=[1,12,12,12,1]
#nNeurons=[numParams,256,128,64,numParams*prediction_target]
#nNeuronsSingle=[1,256,128,64,1]
labbS=['Untrained','Trained'] # additional label to figure headings
ParamFolder='Model_{0:d}_{1:d}_{2:d}_ep{3:d}_hist{4:d}_pred{5:d}'.format(nNeurons[1],nNeurons[2],nNeurons[3],epochs,history_length,prediction_target)+'_TEMP'
if omitForwards:
    ParamFolder = ParamFolder + '_OmitForward'
    for tind,t in enumerate(labbS):
        labbS[tind]=t+'_turnsOnly'
ParamFolder = ParamFolder + '_RNN'
for tind,t in enumerate(labbS):
    labbS[tind]=t+'_RNN'
history_length+=prediction_target  # hack
GroupHistS,GroupModelS=[],[]
# loop through groups
for groupLoop,GroupName in enumerate(GroupNameS):
    # Get tracking and bout files (if they exist) (controls)
    tracking_paths_controls = []
    tracking_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/"+GroupName+"/*tracking.npz")
    bout_paths_controls = []
    bout_paths_controls += glob.glob(DATAROOT + "/GroupedTracking/"+GroupName+"/*bouts.npz")
    # Get tracking and bout files (if they exist)  files (ablation)
    tracking_paths_lesions = []
    #tracking_paths_lesions += glob.glob(DATAROOT + "/GroupedTracking/EA_M0/*tracking.npz")
    bout_paths_lesions = []
    groups=[tracking_paths_controls]#,tracking_paths_lesions]
    boutGroups=[bout_paths_controls]#,bout_paths_lesions]
    
    print('Grabbing and splitting data for RNNs')
    [train_set,train_goal,test_set,test_goal]=AZMo.buildTrainTestSetsRNN(boutGroups,omitForwards=omitForwards,FPS=FPS,prediction_target=prediction_target,history_length=history_length)

    # Collect and prepare data from all fish
    DATALIST=[train_set,test_set]#,train_set_shuffle,test_set_shuffle]
    GOALLIST=[train_goal,test_goal]#,train_goal_shuffle,test_goal_shuffle]
    THRESHOLDS=[100,180,60] # upper limits set on Distance, Angle (abs), IBI 
    
    print('Clipping and Standardising Datasets')
    DATALIST=AZMo.clipAndStandardise(DATALIST,paramLabels,scale=scale,THRESHOLDS=THRESHOLDS)
    
    # generate sin waves with some noise to use as data; same dims and training split as real
    print('Generating sin wave and random datasets to sanity check models')
    aa=DATALIST[0][:,:-1,0].shape
    batch_size=aa[0]
    num_steps=aa[1]
    [xTrain,yTrain,xTest,yTest]=AZMo.generateSinWaveSeries(batch_size,num_steps,addNoise=True,Split=True)
    
    grouped=[xTrain,xTest,yTrain,yTest]
    for i,s in enumerate(grouped):
        grouped[i]=AZMo.makeSeries(s)
    del xTrain,xTest,yTrain,yTest
    [xTrainSin,xTestSin,yTrainSin,yTestSin]=grouped

    ## Generate some random numbers of the same dimensions as data
    a=grouped[0].shape
    xTrainRand=np.random.rand(a[0],a[1],1)
    a=grouped[1].shape
    xTestRand=np.random.rand(a[0],a[1],1)
    xTestRandShuff=xTestRand[np.random.permutation(np.arange(0,xTestRand.shape[0],1)),:,:]
    a=grouped[2].shape
    yTrainRand=np.random.rand(a[0],a[1],1)
    a=grouped[3].shape
    yTestRand=np.random.rand(a[0],a[1],1)
    yTestRandShuff=yTestRand[np.random.permutation(np.arange(0,yTestRand.shape[0],1))]


    # Build and compile multivariate model
    print('Building and compiling multi- and uni-variate RNN models')
    model_RNN = AZMo.build_and_compile_RNNmodel(nNeurons,'mse','adam','mae',LSTM=LSTM,input_shape=[None,3])

    # Build and compile univariate models for each parameter
    model_RNN_Dist = AZMo.build_and_compile_RNNmodel(nNeuronsSingle,'mse','adam','mae',LSTM=LSTM,input_shape=[None,1])
    model_RNN_Angle = AZMo.build_and_compile_RNNmodel(nNeuronsSingle,'mse','adam','mae',LSTM=LSTM,input_shape=[None,1])
    model_RNN_IBI = AZMo.build_and_compile_RNNmodel(nNeuronsSingle,'mse','adam','mae',LSTM=LSTM,input_shape=[None,1])

    # Build and compile univariate model (identical) to test against random numbers and sin waves
    model_RNN_Sin = AZMo.build_and_compile_RNNmodel(nNeuronsSingle,'mse','adam','mae',LSTM=LSTM,input_shape=[None,1])
    model_RNN_Rand = AZMo.build_and_compile_RNNmodel(nNeuronsSingle,'mse','adam','mae',LSTM=LSTM,input_shape=[None,1])

    # Fetch combined data
    xTrainAll=DATALIST[0][:,:-1,:]
    yTrainAll=DATALIST[0][:,-1,:]
    
    # Fetch all test data
    xTestAll=DATALIST[1][:,:-1,:]
    yTestAll=DATALIST[1][:,-1,:]
    xTestDist=DATALIST[1][:,:-1,0]
    yTestDist=DATALIST[1][:,-1,0]
    xTestAngle=DATALIST[1][:,:-1,1]
    yTestAngle=DATALIST[1][:,-1,1]
    xTestIBI=DATALIST[1][:,:-1,2]
    yTestIBI=DATALIST[1][:,-1,2]
    
    for trainLoop,labb in enumerate(labbS):
        
        history_comb,history_Dist,history_Angle,history_IBI,historySin,historyRand=[],[],[],[],[],[]
        
        if trainLoop==1 or labb=='Trained': # Train multivariate model on all data from this group
            print('Training multi-variate model...')
            history_comb=model_RNN.fit(xTrainAll,yTrainAll,epochs=epochs)

            # Train Univariate models on each parameters of same dataset
            # Distance
            xTrain=AZMo.makeSeries(DATALIST[0][:,:-1,0])
            yTrain=AZMo.makeSeries(DATALIST[0][:,-1,0])
            print('Training Univariate Distance model...')
            history_Dist=model_RNN_Dist.fit(xTrain,yTrain,epochs=epochs)
            
            # Angle
            xTrain=AZMo.makeSeries(DATALIST[0][:,:-1,1])
            yTrain=AZMo.makeSeries(DATALIST[0][:,-1,1])
            print('Training Univariate Angle model...')
            history_Angle=model_RNN_Angle.fit(xTrain,yTrain,epochs=epochs)
            
            # IBI
            xTrain=AZMo.makeSeries(DATALIST[0][:,:-1,2])
            yTrain=AZMo.makeSeries(DATALIST[0][:,-1,2])
            print('Training Univariate IBI model...')
            history_IBI=model_RNN_IBI.fit(xTrain,yTrain,epochs=epochs)
                
            # Train Univariate models on random numbers and sin waves
            print('Training Univariate Model on Random Numbers')
            historyRand=model_RNN_Rand.fit(xTrainRand,yTrainRand,epochs=epochs)
            print('Training Univariate Model on Sine Waves')
            historySin=model_RNN_Sin.fit(xTrainSin,yTrainSin,epochs=epochs)

            print('Storing Histories and Models')
            thisGroupHists=[history_comb,history_Dist,history_Angle,history_IBI,historySin,historyRand]
            thisGroupModels=[model_RNN,model_RNN_Dist,model_RNN_Angle,model_RNN_IBI,model_RNN_Sin,model_RNN_Rand]
            GroupModelS.append(thisGroupModels)
            GroupHistS.append(thisGroupHists)
         
         
         
        # Make true and shuffled predictions
        # Multivariate
        print('Making ' + labb + ' Predictions...')
        pred_combined,pred_combinedShuff,yTestShuff,mae_trueS,mae_shuff_pS,mae_shuff_tS = AZMo.testModel(model_RNN,xTestAll,yTestAll,Uni=False)
        
        # Univariate, one by one
        pred_Dist,pred_DistShuff,yTestDistShuff,mae_true_Dist,mae_shuff_Dist_p,mae_shuff_Dist_t = AZMo.testModel(model_RNN_Dist,xTestDist,yTestDist,Uni=True)
        pred_Angle,pred_AngleShuff,yTestAngleShuff,mae_true_Angle,mae_shuff_Angle_p,mae_shuff_Angle_t = AZMo.testModel(model_RNN_Angle,xTestAngle,yTestAngle,Uni=True)
        pred_IBI,pred_IBIShuff,yTestIBIShuff,mae_true_IBI,mae_shuff_IBI_p,mae_shuff_IBI_t = AZMo.testModel(model_RNN_IBI,xTestIBI,yTestIBI,Uni=True)

        # Univariate Sin and random numbers
        predSin,predSinShuff,yTestSinShuff,mae_true_sin,mae_shuff_sin_p,mae_shuff_sin_t = AZMo.testModel(model_RNN_Sin,xTestSin,yTestSin,Uni=True,hack=False)
        predRand,predRandShuff,yTestRandShuff,mae_true_rand,mae_shuff_rand_p,mae_shuff_rand_t = AZMo.testModel(model_RNN_Rand,xTestRand,yTestRand,Uni=True,hack=False)
        
        # Segregate parameters for easy direct comparison
        # predictions
        predDistComb=pred_combined[:,0]
        predAngleComb=pred_combined[:,1]
        predIBIComb=pred_combined[:,2]
        predDistCombShuff=pred_combinedShuff[:,0]
        predAngleCombShuff=pred_combinedShuff[:,1]
        predIBICombShuff=pred_combinedShuff[:,2]
        
        # maes
        mae_true_DistComb=mae_trueS[0]
        mae_true_AngleComb=mae_trueS[1]
        mae_true_IBIComb=mae_trueS[2]
        
        # Real predictions vs predictions based on shuffled inputs
        mae_shuff_DistComb_p=mae_shuff_pS[0]
        mae_shuff_AngleComb_p=mae_shuff_pS[1]
        mae_shuff_IBIComb_p=mae_shuff_pS[2]
        
        # Real predictions vs shuffled targets
        mae_shuff_DistComb_t=mae_shuff_tS[0]
        mae_shuff_AngleComb_t=mae_shuff_tS[1]
        mae_shuff_IBIComb_t=mae_shuff_tS[2]
        
        ### Relative MAEs
        ## Multivariate vs shuffled predictions
        rel_mae_DistComb_p=mae_true_DistComb/mae_shuff_DistComb_p
        rel_mae_AngleComb_p=mae_true_AngleComb/mae_shuff_AngleComb_p
        rel_mae_IBIComb_p=mae_true_IBIComb/mae_shuff_IBIComb_p
        
        ## Multivariate vs shuffled targets
        rel_mae_DistComb_t=mae_true_DistComb/mae_shuff_DistComb_t
        rel_mae_AngleComb_t=mae_true_AngleComb/mae_shuff_AngleComb_t
        rel_mae_IBIComb_t=mae_true_IBIComb/mae_shuff_IBIComb_t
        
        ## Univariate vs shuffled predictions
        rel_mae_Dist_p=mae_true_Dist/mae_shuff_Dist_p
        rel_mae_Angle_p=mae_true_Angle/mae_shuff_Angle_p
        rel_mae_IBI_p=mae_true_IBI/mae_shuff_IBI_p
        
        ## Univariate vs shuffled targets
        rel_mae_Dist_t=mae_true_Dist/mae_shuff_Dist_t
        rel_mae_Angle_t=mae_true_Angle/mae_shuff_Angle_t
        rel_mae_IBI_t=mae_true_IBI/mae_shuff_IBI_t
        
        # Univariate random numbers
        rel_mae_Rand_p=mae_true_rand/mae_shuff_rand_p
        rel_mae_Rand_t=mae_true_rand/mae_shuff_rand_t
        
        # Univariate sin wave
        rel_mae_sin_p=mae_true_sin/mae_shuff_sin_p
        rel_mae_sin_t=mae_true_sin/mae_shuff_sin_t
        
        # print all results
        # Sin wave vs predictions based on shuffled inputs
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Random Numbers vs shuffled actual = ' + str(rel_mae_Rand_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Random Numbers vs shuffled inputs = ' + str(rel_mae_Rand_t))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Sine Wave vs shuffled actual = ' + str(rel_mae_sin_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Since Wave vs shuffled inputs = ' + str(rel_mae_sin_t))
        
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate Distance vs shuffled actual = ' + str(rel_mae_DistComb_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate Distance vs shuffled inputs  = ' + str(rel_mae_DistComb_t))
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate Angle vs shuffled actual  = ' + str(rel_mae_AngleComb_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate Angle vs shuffled inputs  = ' + str(rel_mae_AngleComb_t))
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate IBI vs shuffled actual  = ' + str(rel_mae_IBIComb_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Multivariate IBI vs shuffled inputs  = ' + str(rel_mae_IBIComb_t))
        
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Distance vs shuffled actual = ' + str(rel_mae_Dist_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Distance vs shuffled inputs = ' + str(rel_mae_Dist_t))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Angle vs shuffled actual = ' + str(rel_mae_Angle_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate Angle vs shuffled inputs = ' + str(rel_mae_Angle_t))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate IBI vs shuffled actual = ' + str(rel_mae_IBI_p))
        print(GroupName + ' ' + labb + ' Relative MAE, Univariate IBI vs shuffled inputs = ' + str(rel_mae_IBI_t))

#    pointsize=5
    ### plot predictions against true and shuffled
        if plot:
            if saveFig:
                saveDir=DATAROOT + '\\ModelSummary\\' + ParamFolder + '\\'
                AZU.cycleMkDir(saveDir)
            if subplot:
                ## Mulitvariate
                plt.figure()
                plt.get_current_fig_manager().window.showMaximized()
                #plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
                AZMo.plotRNNMeasure([1,4,2],'MultiVariate_Distance',predDistComb,predDistCombShuff,yTestDist,yTestDistShuff,MetLab='Dist (pix)',ss=pointsize)
                AZMo.plotRNNMeasure([1,4,3],'MultiVariate_Angle',predAngleComb,predAngleCombShuff,yTestAngle,yTestAngleShuff,MetLab='Angle (deg)',ss=pointsize)
                AZMo.plotRNNMeasure([1,4,4],'MultiVariate_IBI',predIBIComb,predIBICombShuff,yTestIBI,yTestIBIShuff,MetLab='IBI (s)',ss=pointsize,legend=True)
                plt.suptitle((GroupName + ' ' + labb + ' Multivariate model prediction accuracy'))
                plt.subplot(1,4,1)
                plt.axis('off')
                saveName=saveDir+GroupName + labb + 'Multivariate_Summary.png'
                plt.savefig(saveName,dpi=600)
            else:
                textstr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeurons[0],nNeurons[1],nNeurons[2],nNeurons[3],nNeurons[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_DistComb_p) 
                AZMo.plotRNNMeasureSepFigs('Multivariate_ Distance'+'_'+GroupName+'_'+labb,predDistComb,predDistCombShuff,yTestDist,yTestDistShuff,textstr,MetLab='Dist (pix)',ss=pointsize,saveFig=saveFig,keepFigures=keepFigures,saveDir=saveDir)
                textstr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeurons[0],nNeurons[1],nNeurons[2],nNeurons[3],nNeurons[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_AngleComb_p)
                AZMo.plotRNNMeasureSepFigs('Multivariate_Angle'+'_'+GroupName+'_'+labb,predAngleComb,predAngleCombShuff,yTestAngle,yTestAngleShuff,textstr,MetLab='Angle (deg)',ss=pointsize,saveFig=saveFig,keepFigures=keepFigures,saveDir=saveDir)
                textstr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeurons[0],nNeurons[1],nNeurons[2],nNeurons[3],nNeurons[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_IBIComb_p)
                AZMo.plotRNNMeasureSepFigs('Multivariate_IBI'+'_'+GroupName+'_'+labb,predIBIComb,predIBICombShuff,yTestIBI,yTestIBIShuff,textstr,MetLab='IBI (s)',ss=pointsize,legend=True,saveFig=saveFig,keepFigures=keepFigures,saveDir=saveDir)

            figname=GroupName + '_' + labb + '_MultivariateRNN_Summary.png'
            if keepFigures==False: plt.close()
            ## Univariate
            if subplot:
                plt.figure()
                plt.get_current_fig_manager().window.showMaximized()
                AZMo.plotRNNMeasure([2,3,4],'Distance',pred_Dist,pred_DistShuff,yTestDist,yTestDistShuff,MetLab='Distance (pix)',ss=pointsize)
                AZMo.plotRNNMeasure([2,3,5],'Angle',pred_Angle,pred_AngleShuff,yTestAngle,yTestAngleShuff,MetLab='Angle (deg)',ss=pointsize)
                AZMo.plotRNNMeasure([2,3,6],'IBI',pred_IBI,pred_IBIShuff,yTestIBI,yTestIBIShuff,MetLab='IBI (s)',ss=pointsize)
                AZMo.plotRNNMeasure([2,3,2],'Random_Numbers',predRand,predRandShuff,yTestRand,yTestRandShuff,ss=pointsize)
                AZMo.plotRNNMeasure([2,3,3],'Sin_Wave',predSin,predSinShuff,yTestSin,yTestSinShuff,ss=pointsize,legend=True)
                plt.suptitle(GroupName + ' ' + labb + ' Univariate Model Prediction accuracy')
                plt.subplot(2,3,1)
                plt.axis('off')
            else:
                textStr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeuronsSingle[0],nNeuronsSingle[1],nNeuronsSingle[2],nNeuronsSingle[3],nNeuronsSingle[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_Dist_p)
                saveDir=DATAROOT + '\\ModelSummary\\' + ParamFolder + '\\'
                AZMo.plotRNNMeasureSepFigs('Univariate_Distance'+'_'+GroupName+'_'+labb,pred_Dist,pred_DistShuff,yTestDist,yTestDistShuff,textStr,saveDir=saveDir,MetLab='Distance (pix)',ss=pointsize,saveFig=saveFig,keepFigures=keepFigures)
                textStr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeuronsSingle[0],nNeuronsSingle[1],nNeuronsSingle[2],nNeuronsSingle[3],nNeuronsSingle[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_Angle_p)
                AZMo.plotRNNMeasureSepFigs('Univariate_Angle'+'_'+GroupName+'_'+labb,pred_Angle,pred_AngleShuff,yTestAngle,yTestAngleShuff,textStr,saveDir=saveDir,MetLab='Angle (deg)',ss=pointsize,saveFig=saveFig,keepFigures=keepFigures)
                textStr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeuronsSingle[0],nNeuronsSingle[1],nNeuronsSingle[2],nNeuronsSingle[3],nNeuronsSingle[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_IBI_p)
                AZMo.plotRNNMeasureSepFigs('Univariate_IBI'+'_'+GroupName+'_'+labb,pred_IBI,pred_IBIShuff,yTestIBI,yTestIBIShuff,textStr,saveDir=saveDir,MetLab='IBI (s)',ss=pointsize,saveFig=saveFig,keepFigures=keepFigures)
                textStr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeuronsSingle[0],nNeuronsSingle[1],nNeuronsSingle[2],nNeuronsSingle[3],nNeuronsSingle[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_Rand_p)
                AZMo.plotRNNMeasureSepFigs('Univariate_Random_Numbers'+'_'+GroupName+'_'+labb,predRand,predRandShuff,yTestRand,yTestRandShuff,textStr,saveDir=saveDir,ss=pointsize,saveFig=saveFig,keepFigures=keepFigures)
                textStr = 'Model Layers = {0:d},{1:d},{2:d},{3:d},{4:d}'.format(nNeuronsSingle[0],nNeuronsSingle[1],nNeuronsSingle[2],nNeuronsSingle[3],nNeuronsSingle[4]) + '\nepochs={0:d}'.format(epochs) + '\nhistory={0:d}'.format(history_length-1) + '\nRMAE={0:2f}'.format(rel_mae_sin_p)
                AZMo.plotRNNMeasureSepFigs('Univariate_Sin_Wave'+'_'+GroupName+'_'+labb,predSin,predSinShuff,yTestSin,yTestSinShuff,textStr,saveDir=saveDir,ss=pointsize,legend=True,saveFig=saveFig,keepFigures=keepFigures)
            
            
            
            if keepFigures==False: plt.close()
# END