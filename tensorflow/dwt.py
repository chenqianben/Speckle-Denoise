#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers

import numpy as np


'''discret wavelet transform 2D(haar)'''
def dwt_init(x):
    x01 = x[:,0::2,:,:]/2
    x02 = x[:,1::2,:,:]/2
    
    x1 = x01[:,:,0::2,:]
    x2 = x01[:,:,1::2,:]
    x3 = x02[:,:,0::2,:]
    x4 = x02[:,:,1::2,:]
    
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    
    return tf.concat([x_LL, x_HL, x_LH, x_HH], axis=-1)

class DWT(Model):
    def __init__(self):
        super(DWT, self).__init__()
        #self.requires_grad = False
        
    def call(self, x):
        return dwt_init(x)




'''inverse discret wavelet transform 2D(haar)'''
def idwt_init(x):
    r = 2
    in_bs, in_h, in_w, in_c = x.shape
    out_bs, out_h, out_w, out_c = in_bs, r*in_h, r*in_w, int(in_c/(r**2))
    
    x1 = x[:,:,:,0:out_c]/2
    x2 = x[:,:,:,out_c:out_c*2]/2
    x3 = x[:,:,:,out_c*2:out_c*3]/2
    x4 = x[:,:,:,out_c*3:out_c*4]/2
           
    h01 = x1 - x2 - x3 + x4
    h02 = x1 + x2 - x3 - x4
    h03 = x1 - x2 + x3 - x4
    h04 = x1 + x2 + x3 + x4
    
    h01 = tf.split(h01, h01.shape[1], 1)
    h02 = tf.split(h02, h02.shape[1], 1)
    h12 = []
    for i in range(in_h):
        h12.append(h01[i])
        h12.append(h02[i])
    h12 = tf.concat(h12, axis=1)
        
    h03 = tf.split(h03, h03.shape[1], 1)
    h04 = tf.split(h04, h04.shape[1], 1)
    h34 = []
    for i in range(in_h):
        h34.append(h03[i])
        h34.append(h04[i])
    h34 = tf.concat(h34, axis=1)
    
    h = []
    for i in range(in_w):
        h.append(h12[:,:,i,:])
        h.append(h34[:,:,i,:])
    
    h = tf.stack(h, axis=2)
    return h


class IDWT(Model):
    def __init__(self):
        super(IDWT, self).__init__()
        #self.requires_grad = False
        
    def call(self, x):
        return idwt_init(x)

