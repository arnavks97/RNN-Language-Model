#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:14:20 2018

@author: arnav
"""

import numpy as np


def softmax(y):
    yT = np.exp(y - np.max(y))
    return yT/np.sum(yT)


def save_param(output, model):
    param1 = model.param1.get_value()
    param2 = model.param2.get_value()
    param3 = model.param3.get_value()
    
    np.savez(output, param1 = param1, param2 = param2, param3 = param3)
    
    
def load_param(path, model):
    f = np.load(path)
    param1 = f["param1"]
    param2 = f["param2"]
    param3 = f["param3"]
    
    model.hidden_size = param1.shape[0]
    model.vocab_size = param1.shape[1]
    
    model.param1.set_value(param1)
    model.param2.set_value(param2)
    model.param3.set_value(param3)
    
    print "Vocabulary size = %d" (param1.shape[1])
    print "Hidden layer size = %d" (param1.shape[0])