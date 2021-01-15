# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:57:28 2020

@author: Tom
"""
import numpy as np
import math

# Define a gaussian function with offset
def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

# Define exponential function with offset
def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

# rotate a point around origin by angle in degrees            
def rotatePointAboutOrigin(x,y,angle):
    
    radians=math.radians(angle)
    qx = math.cos(radians) * (x) - math.sin(radians) * (y)
    qy = math.sin(radians) * (x) + math.cos(radians) * (y)
        
    return qx,qy

def smoothSignal(x,N=5):
## performs simple 'box-filter' of any signal, with a defined box kernel size
    xx=np.convolve(x, np.ones((int(N),))/int(N), mode='valid')
    n=N-1
    xpre=np.zeros(n)
    xxx=np.concatenate((xpre,xx))
    return xxx