# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:04:18 2020

@author: thoma
"""
lib_path =r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
import sys
sys.path.append(lib_path)
import glob
import pandas as pd  
import AZ_utilities as AZU

def printMovieListFolderList(folderListFile,p=True):
    _, folderNames = AZU.read_folder_list(folderListFile)
    list2=[]
    for folder in folderNames:
        temp_list=printMovieList(folder,list2)
        if list2==[]:
            for n in temp_list:
                list2.append(n)
    if p:
        df = pd.DataFrame(list2)
        fileName=folder+r'\\AllMovieList.csv'
        df.to_csv(fileName, index=False)
        print(folder)
    return list2

def printMovieList(folder,list2=[],p=True):
    
    list1=glob.glob(folder+'*.avi')
    
    for li in list1:
        list2.append(li.rsplit(sep='\\',maxsplit=1)[1][0:-4])
    if p:
        df = pd.DataFrame(list2)
        fileName=folder+r'\\MovieList.csv'
        df.to_csv(fileName, index=False)
        
    return list2