#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.feature import hog
from skimage.exposure import adjust_gamma
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans 
from skimage.filters import gaussian, median, laplace
from skimage.draw import ellipse_perimeter

from math import ceil, floor
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import pydicom

from read_roi import read_roi_file, read_roi_zip

def read_mri_data(root_path, if_normalized=True):
    '''read five types of images from the parent folder, if_normalized points to the T1 SAG images'''
    ims_T1s = []
    ims_T1 = []
    ims_T2s = []
    ims_T2 = []
    ims_T2st = []
    pos = []
    axis_ens_T1 = []
    axis_ens_T2 = []
    axis_ens_T2star = []
    
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if root.find('T1_TSE_SAG') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T1s = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T1s = (im_T1s-im_T1s.mean())/im_T1s.std()
                    ims_T1s.append(im_T1s)
                    pos.append(os.path.abspath(os.path.dirname(root)))
                    break
                    
            if root.find('T1_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2)
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T1 = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T1 = (im_T1-im_T1.mean())/im_T1.std()
                    ims_T1.append(im_T1)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T1 = np.array([xs,ys]).T
                    axis_ens_T1.append(axis_T1)
                    
            if root.find('T2_TSE_SAG') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2s = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2s = (im_T2s-im_T2s.mean())/im_T2s.std()
                    ims_T2s.append(im_T2s)
                    break
                    
            if root.find('T2_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2 = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2 = (im_T2-im_T2.mean())/im_T2.std()
                    ims_T2.append(im_T2)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T2 = np.array([xs,ys]).T
                    axis_ens_T2.append(axis_T2)
                    
            if root.find('T2Star_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2st = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2st = (im_T2st-im_T2st.mean())/im_T2st.std()
                    ims_T2st.append(im_T2st)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T2star = np.array([xs,ys]).T
                    axis_ens_T2star.append(axis_T2star)
                    
    return np.array(ims_T1s), np.array(ims_T1), np.array(ims_T2s), np.array(ims_T2), np.array(ims_T2st), pos, np.array(axis_ens_T1), np.array(axis_ens_T2), np.array(axis_ens_T2star)

def read_imagenet_data(root_path):
    ims = []
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            im = io.imread(os.path.join(root,filename), as_gray=False)
            ims.append(im)
    return np.array(ims)

def normalize(ims):
    """normalization to 0~1"""
    ims_new = []
    for im in ims:
        ims_new.append((im-im.min())/(im.max()-im.min()))
    return np.array(ims_new).astype(np.float32)

def add_noise(ims, mean=0, var=1e-2, n_type='gaussian', seed=42):
    np.random.seed(seed)
    ims = ims.astype(np.float32)
    if n_type == "gaussian":
        num_ims,row,col,ch= ims.shape
        sigma = var**0.5
        gauss = np.stack([np.random.normal(mean, sigma, (row,col,ch)) for _ in range(num_ims)]).astype(np.float32)
        gauss = gauss.reshape(num_ims,row,col,ch)
        noisy = ims + gauss
        return noisy

    elif n_type =="speckle":
        num_ims,row,col,ch = ims.shape
        gauss = np.stack([np.random.normal(0, 1, (row,col,ch)) for _ in range(num_ims)]).astype(np.float32)
        gauss = gauss.reshape(num_ims,row,col,ch)  
        noisy = ims + ims * gauss
        return noisy