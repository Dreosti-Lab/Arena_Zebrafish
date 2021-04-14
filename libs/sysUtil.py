# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:46:56 2020

@author: thoma
"""
#
import shelve,glob

def shelveWork(filename=[]):

    wDir=r'D:\\Shelf\\'
    if(len(filename)==0):
        d=len(glob.glob(wDir+'*.out'))
        name=wDir + 'shelved_' + str(d) + '.out'
        print('Shelving work here: ' + filename)
    else:
        d=len(glob.glob(wDir+filename+'*.out'))
        filename=wDir+filename+'_'+str(d)+'.out'
        print('Shelving work here: ' + filename)
        my_shelf = shelve.open(filename,'n') # 'n' for new
    
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()
    
def restoreShelf(filename=[]):
    
    if(len(filename)==0):
        print('Please provide the name of the shelved session')
    else:
        filename='D:\\Shelf\\'+filename+'.out'
        my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
