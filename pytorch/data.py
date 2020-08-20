#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
import sys

import skimage.io as io

from utils import read_imagenet_data, add_noise, normalize

root_path = r'..\ImageNet_database'
ims = read_imagenet_data(root_path)
ims = normalize(ims[:,:,:,np.newaxis])
ims_noise = add_noise(ims, mean=0, var=1e-3, n_type='gaussian')
ims_noise = normalize(ims_noise)




