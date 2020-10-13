# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:57:28 2020

@author: Tom
"""
import numpy as np

# Define a gaussian function with offset
def gaussian_func(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))