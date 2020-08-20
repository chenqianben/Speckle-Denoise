#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
import sys

import skimage.io as io
from utils import read_imagenet_data, read_div2k_data, read_ultrasound_data
from utils import images2patches, normalize, add_noise

def read_data(type_ims):
    if type_ims == 'imagenet':
        root_path = r'../ImageNet_database'
        ims = read_imagenet_data(root_path)
        ims = normalize(ims[:,:,:,np.newaxis])
        #ims_noise = add_noise(ims, mean=0, var=1e-3, n_type='gaussian')
        #ims_noise = normalize(ims_noise)
        return ims
    
    elif type_ims == 'div2k':
        size = (128,128)
        
        train_noise_maps = [0.001, 0.005, 0.01, 0.05, 0.1]
        test_noise_maps = [0.5]
        
        train_X, train_Y = read_div2k_data("../DIV2K_database/DIV2K_train_HR", if_normalized=True)
        test_X, test_Y = read_div2k_data("../DIV2K_database/DIV2K_valid_HR", if_normalized=True)
        
        train_X_p = images2patches(train_X, size, train_noise_maps)
        train_Y_p = images2patches(train_Y, size)
        test_X_p = images2patches(test_X, size, test_noise_maps)
        test_Y_p = images2patches(test_Y, size)
        return (train_X_p, train_Y_p), (test_X_p, test_Y_p)
    
    elif type_ims == 'ultrasound':
        size = (128,128)
        train_X, train_Y = read_ultrasound_data("../Ultrasound_database/train", if_normalized=True)
        test_X, test_Y = read_ultrasound_data("../Ultrasound_database/test", if_normalized=True)
        
        padding = ((0,0),(0,10))
        train_X_p = images2patches(train_X, size, padding=padding)
        train_Y_p = images2patches(train_Y, size, padding=padding)
        test_X_p = images2patches(test_X, size, padding=padding)
        test_Y_p = images2patches(test_Y, size, padding=padding)
        return (train_X_p, train_Y_p), (test_X_p, test_Y_p)
        
    else:
        print("Parameter 'type_ims' should be 'imagenet', 'div2k' or 'ultrasound'")
        return




