#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:48:33 2018

@author: therealrussellkim
"""

import numpy as np

def Boltzmann(belief,theta,iteration):
    U = np.random.uniform(0,1)
    argMax = 0
    maxMu = max(belief)
    length = len(belief)
    p_x = [0 for i in range(length)]
    P_x = [0 for i in range(length)]
    for i in range(length):
        p_x[i] = np.exp(theta*(belief[i] - maxMu))
    sumPx = sum(p_x)
    for i in range(length):
        p_x[i] = p_x[i]/sumPx
        #Get the cumulative probability 
        if i == 1:
            P_x[i] = p_x[i]
        else:
            P_x[i] = P_x[i-1]+p_x[i]
        #Check if cumulative distribution goes over U 
        if P_x[i] <= U:
            argMax = i
    return(argMax)