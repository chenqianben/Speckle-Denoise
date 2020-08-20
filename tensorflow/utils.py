#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf

from math import ceil, floor
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
import pydicom

from model_noise_est import FCN
from read_roi import read_roi_file, read_roi_zip

'''Imagenet Data'''
def read_imagenet_data(root_path):
    ims = []
    n = 0
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            try:
                im = io.imread(os.path.join(root,filename), as_gray=False)
                ims.append(im)
            except Exception as e:
                n += 1
    print(n, 'images loading failed')
    return np.array(ims)

'''div2k Data'''
def read_div2k_data(fname, num=100, if_normalized=True):
    X_fname = os.path.join(fname, "images_myspeckled")
    Y_fname = os.path.join(fname, "images_myoriginal")

    X_data = []
    Y_data = []
    
    # read X data
    n1 = 0
    for root, dirnames, filenames in os.walk(X_fname):
        for filename in filenames:
            n1 += 1
            im = io.imread(os.path.join(root,filename), as_gray=False).astype(np.float32)
            im = (im-im.min())/(im.max()-im.min()) if if_normalized else im
            X_data.append((int(filename[:4]), im))
            if n1 >= num:
                break

    # read Y data
    n2 = 0
    for root, dirnames, filenames in os.walk(Y_fname):
        for filename in filenames:
            n2 += 1
            im = io.imread(os.path.join(root,filename), as_gray=False).astype(np.float32)
            if len(im.shape) == 3:
                #im = im.mean(axis=-1)
                im = im[:,:,0]*0.2989 + im[:,:,1]*0.5870 + im[:,:,2]*0.1140 
            im = (im-im.min())/(im.max()-im.min()) if if_normalized else im
            Y_data.append((int(filename[:4]), im))
            if n2 >= num:
                break     
    
    print('Totally', n1,'images loaded for X and',n2,'images loaded for Y')
    return X_data, Y_data

'''Philips ultrasound Data'''
def read_ultrasound_data(fname, if_normalized=True):
    X_data = []
    Y_data = []
    
    # read X data
    n1 = n2 = 0
    for root, dirnames, filenames in os.walk(fname):
        for filename in sorted(filenames):
            #print(filename)
            if filename.endswith('desp.png'):
                im = io.imread(os.path.join(root,filename), as_gray=False).astype(np.float32)
                im = (im-im.min())/(im.max()-im.min()) if if_normalized else im
                Y_data.append((int(filename[:4]), im))
                n1 += 1
            else:
                im = io.imread(os.path.join(root,filename), as_gray=False).astype(np.float32)
                im = (im-im.min())/(im.max()-im.min()) if if_normalized else im
                X_data.append((int(filename[:4]), im))
                n2 += 1
    
    print('Totally', n1,'images loaded for X and', n2,'images loaded for Y')
    return X_data, Y_data



def images2patches(ims, size=(128,128), noise_maps=False, padding=((0,0),(0,0))):
    N = len(ims)
    patches = []
    if noise_maps:
        N_per_map = N//len(noise_maps)
        
    for i in range(N):
        ind = ims[i][0]
        im = ims[i][1]
        im = skimage.util.pad(im, padding, 'reflect')
        
        cur_patches = []
        n_hor, n_ver = im.shape[0]//size[0], im.shape[1]//size[1]
        for j in range(n_hor):
            for k in range(n_ver):
                cur_patch = im[j*size[0]:(j+1)*size[0], k*size[1]:(k+1)*size[1]]
                if noise_maps:
                    cur_ind = (ind-1)//N_per_map if ind <= 800 else (ind-801)//N_per_map
                    cur_noise_map = (noise_maps[cur_ind]*np.ones(size)*cur_patch).astype(np.float32)
                    cur_patch = np.stack((cur_patch, cur_noise_map), axis=-1)
                cur_patches.append(cur_patch)    
        cur_patches = np.stack(cur_patches, axis=0)
        patches.append((ind, cur_patches))
    return patches
        
'''Auxiliary functions'''
def normalize(ims):
    """normalization to 0~1"""
    assert len(ims.shape) == 4
    ims_new = []
    for im in ims:
        ims_new.append((im-im.min())/(im.max()-im.min()+1e-12))
    return np.array(ims_new).astype(np.float32)

def add_noise(ims, mean=0, var=100, n_type='gaussian', seed=42):
    np.random.seed(seed)
    ims = ims.astype(np.float32)
    if n_type == "gaussian":
        num_ims,row,col,ch= ims.shape
        sigma = var**0.5
        gauss = np.stack([np.random.normal(mean, sigma, (row,col,ch)) for _ in range(num_ims)]).astype(np.float32)
        gauss = gauss.reshape(num_ims,row,col,ch)
        #noisy = np.clip(ims + gauss, 0, 255)
        noisy = ims + gauss
        return noisy

    elif n_type =="speckle":
        num_ims,row,col,ch = ims.shape
        gauss = np.stack([np.random.normal(0, 1, (row,col,ch)) for _ in range(num_ims)]).astype(np.float32)
        gauss = gauss.reshape(num_ims,row,col,ch)  
        #noisy = np.clip(ims + ims * gauss, 0, 255)
        noisy = ims + ims * gauss
        return noisy
    
def add_noise_est(ims, if_est = False, var=0.05):
    std = np.sqrt(var)
    bs, h, w, c = ims.shape
    if not if_est:
        ims_with_est = np.concatenate([ims, std * np.ones((bs, h, w, c))], axis=-1).astype(np.float32)
    else:
        model = FCN(color = False, channels = [16, 16, 32, 32, 64, 64, 32, 32, 16, 16],
                    channel_att=False, spatial_att=False, use_bias = True)
        model.load_weights(filepath = "model_weights/model_noise_est.ckpt")
        
        batch_size = 16
        test_Y = np.zeros_like(ims)
        test_dataset = tf.data.Dataset.from_tensor_slices((ims,test_Y))
        test_dataset = test_dataset.batch(batch_size).prefetch(1)
        
        pred_Y = []
        for (batch_test_X, batch_test_Y) in test_dataset: 
            pred_test_Y = model(batch_test_X)
            pred_Y.append(pred_test_Y.numpy())
        pred_Y = np.concatenate(pred_Y)
        
        ims_with_est = np.concatenate([ims, pred_Y], axis=-1).astype(np.float32)
    return ims_with_est

def squeeze_patches(patches):
    patches_squeeze = []
    labels = []
    for (label, cur_patches) in patches:
        labels.append([label]*cur_patches.shape[0])
        patches_squeeze.append(cur_patches)
    labels = [item for sublist in labels for item in sublist]
    patches_squeeze = np.concatenate(patches_squeeze, axis=0)
    return patches_squeeze, labels

