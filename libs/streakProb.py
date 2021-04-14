# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:24:27 2020

@author: thoma
"""

import numpy as np

def angleToSeq(angles):
    
    turnThresh=10 # threshold for a turn in degrees
    seq=[]        
    for i in angles:
        if i >turnThresh: seq.append('L')
        elif i <(turnThresh*-1): seq.append('R')
        elif i <turnThresh and i >(turnThresh*-1): seq.append('F')
        
    return seq
# finds the sumulative probability of finding a given sequence as a function of number of consecutive bouts
def permCumulativeProb(boo,perm,nn=50):
    
    pj=perm[0]
    pk=perm[1]
    pl=perm[2]
    probs=np.zeros(nn)
    
    for booLen in range(len(perm),nn):      #   cycle through tested sequence lengths
        count=0
        count_bl=0
        for i in range(0,len(boo)-booLen):         #   start at every point in the bout list...      
            count_bl+=1
            for k in range(i,i+booLen-1):     #   cycle through indices for all segments of nn length from point i
                if boo[k]==pj and boo[k+1]==pk and boo[k+2]==pl:
                    count+=1    #   count how many times this sequence is found in bout sequences this long
                    break
        probs[booLen]=count/count_bl
            
    return probs
        

### finds the relative frequency of finding every permutation of the provided sequence in pairs. Default is forward left and right turns ['F','L','R'], but can be any list of strings
def probSeq1(boo,pot=['F','L','R']):
    
    probs=[]
    combs=[]
        
    for j in range(0,len(pot)):
        for k in range(0,len(pot)):
            pj=pot[j]
            pk=pot[k]
            count=0 # reset count
            for i in range(0,len(boo)-len(pot)):   # cycle through bout list (exclude last as start of pair)
                if boo[i]==pj and boo[i+1]==pk:
                    count+=1
            probs.append(count/(len(boo)-len(pot)))
            combs.append(pj+pk)
    return combs,probs


### finds the relative frequency of finding every permutation of the provided sequence in triplets. Default is forward left and right turns ['F','L','R'], but can be any list of strings
def probSeq(boo,pot=['F','L','R']):
    
    probs=[]
    combs=[]
        
    for j in range(0,len(pot)):
        for k in range(0,len(pot)):
            for l in range(0,len(pot)):
                pj=pot[j]
                pk=pot[k]
                pl=pot[l] # new sequence set
                count=0 # reset count
                for i in range(0,len(boo)-len(pot)):   # cycle through bout list (exclude last two as starts of triplets)
                    if boo[i]==pj and boo[i+1]==pk and boo[i+2]==pl:
                        count+=1
                probs.append(count/i)
                combs.append(pj+pk+pl)
    return combs,probs


def probStreak(boo):
    
    nLa=0
    nRa=0
    cLa1=0
    cRa1=0
    cLa2=0
    cRa2=0
    cLa3=0
    cRa3=0
    cLa4=0
    cRa4=0
    cLa5=0
    cRa5=0
    cLa6=0
    cRa6=0
    cLa7=0
    cRa7=0
    cLa8=0
    cRa8=0
    cLa9=0
    cRa9=0
    cLa10=0
    cRa10=0
    cLa11=0
    cRa11=0
    cLa12=0
    cRa12=0
    
    for i,t in enumerate(boo):
        if t:
            nLa+=1
            if boo[i-1]:
                cLa1+=1
                if boo[i-2]:
                    cLa2+=1
                    if boo[i-3]:
                        cLa3+=1
                        if boo[i-4]:
                            cLa4+=1
                            if boo[i-5]:
                                cLa5+=1
                                if boo[i-6]:
                                    cLa6+=1
                                    if boo[i-7]:
                                        cLa7+=1
                                        if boo[i-8]:
                                            cLa8+=1
                                            if boo[i-9]:
                                                cLa9+=1
                                                if boo[i-10]:
                                                    cLa10+=1
                                                    if boo[i-11]:
                                                        cLa11+=1
                                                        if boo[i-12]:
                                                            cLa12+=1
        else:
            nRa+=1
            if boo[i-1]==False:
                cRa1+=1
                if boo[i-2]==False:
                    cRa2+=1
                    if boo[i-3]==False:
                        cRa3+=1
                        if boo[i-4]==False:
                            cRa4+=1
                            if boo[i-5]==False:
                                cRa5+=1
                                if boo[i-6]==False:
                                    cRa6+=1
                                    if boo[i-7]==False:
                                        cRa7+=1
                                        if boo[i-8]==False:
                                            cRa8+=1
                                            if boo[i-9]==False:
                                                cRa9+=1
                                                if boo[i-10]==False:
                                                    cRa10+=1
                                                    if boo[i-11]==False:
                                                        cRa11+=1
                                                        if boo[i-12]==False:
                                                            cRa12+=1
    cRpRa1=cRa1/nRa
    cRpRa2=cRa2/nRa
    cRpRa3=cRa3/nRa
    cRpRa4=cRa4/nRa
    cRpRa5=cRa5/nRa
    cLpLa5=cLa5/nLa
    cLpLa4=cLa4/nLa
    cLpLa3=cLa3/nLa
    cLpLa2=cLa2/nLa
    cLpLa1=cLa1/nLa
    cRpRa6=cRa6/nRa
    cLpLa6=cLa6/nLa
    cRpRa7=cRa7/nRa
    cLpLa7=cLa7/nLa
    cRpRa8=cRa8/nRa
    cLpLa8=cLa8/nLa
    cRpRa9=cRa9/nRa
    cLpLa9=cLa9/nLa
    cRpRa10=cRa10/nRa
    cLpLa10=cLa10/nLa
    cRpRa11=cRa11/nRa
    cLpLa11=cLa11/nLa
    cRpRa12=cRa12/nRa
    cLpLa12=cLa12/nLa
    
    ca1=(cRpRa1+cLpLa1)/2
    ca2=(cRpRa2+cLpLa2)/2
    ca3=(cRpRa3+cLpLa3)/2
    ca4=(cRpRa4+cLpLa4)/2
    ca5=(cRpRa5+cLpLa5)/2
    ca6=(cRpRa6+cLpLa6)/2
    ca7=(cRpRa7+cLpLa7)/2
    ca8=(cRpRa8+cLpLa8)/2
    ca9=(cRpRa9+cLpLa9)/2
    ca10=(cRpRa10+cLpLa10)/2
    ca11=(cRpRa11+cLpLa11)/2
    ca12=(cRpRa12+cLpLa12)/2
    ca=[ca1,ca2,ca3,ca4,ca5,ca6,ca7,ca8,ca9,ca10,ca11,ca12]
    
    return ca