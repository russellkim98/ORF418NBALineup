#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:48:33 2018

@author: therealrussellkim
"""
import numpy as np
def UCB(belief,theta,iteration,num_selected):
    bonus = [0 for i in range(len(belief))]
    
    for i in range(len(belief)):

        bonus[i] = theta*np.sqrt(np.log(iteration)/num_selected[i])
    bonus_theta = [bonus[i]*theta for i in range(len(bonus))]
    argMax = np.argmax(belief + bonus_theta)
    num_selected[argMax] += 1
    return argMax,num_selected