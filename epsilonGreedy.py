#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:26:56 2018

@author: therealrussellkim
"""

import numpy as np

def EpsilonGreedy(belief,c,iteration):
    tempValue = np.random.uniform(0,1)
    epsilon = c/(iteration+1)
    choice = 0
    # explore with prob = epsilon
    if tempValue < epsilon:
        choice = np.random.randint(0,len(belief)-1)
    # otherwise, exploit
    else:
        choice = np.argmax(belief)

    return(choice)
        