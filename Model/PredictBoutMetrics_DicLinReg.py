# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:17:07 2021

@author: thoma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

GroupDictionary_controlM0 = np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EC_M0.npy',allow_pickle=True).item()
GroupDictionary_ablatedM0 = np.load('D:/Movies/DataForAdam/DataForAdam/GroupDictionaries/EA_M0.npy',allow_pickle=True).item()

groups = [GroupDictionary_controlM0, GroupDictionary_ablatedM0]

group_IBI=[]
group_dSpace=[]
group_dAngle=[]
# loop through groups
for group in groups:
#   cycle through individual fish and collect IBIs
    group_IBI_temp=[]
    for fish in group['Ind_fish']:
        group_IBI_temp.append(np.diff(fish['data']['boutStarts']))
    
    group_IBI.append(group_IBI_temp)    
    group_dSpace.append(group['PooledData']['boutDists'])
    group_dAngle.append(group['PooledData']['boutAngles'])
    
    flat_IBI=[item for sublist in group_IBI[0] for item in sublist]
    flat_dAngle=[item for sublist in group_dAngle[0][0:-1] for item in sublist]
    flat_dSpace=[item for sublist in group_dSpace[0][0:-1] for item in sublist]
    
    
    
    df = pd.DataFrame({"boutDists": , "boutAngles": [item for sublist in group_dAngle[0:-1] for item in sublist], "boutIBI": [item for sublist in group_IBI for item in sublist]})
    
#    df.head()

