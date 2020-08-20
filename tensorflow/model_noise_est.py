#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model, layers


# # model

# In[2]:


"""Defining the block"""
class Basic(Model): 
    def __init__(self, out_ch, g=16, channel_att=False, spatial_att=False, use_bias = True,
                 init = tf.keras.initializers.glorot_normal()):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.conv_block = keras.Sequential([
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, use_bias=use_bias),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, use_bias=use_bias),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, use_bias=use_bias),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])
            
        
        if channel_att: # 提炼重组一下各个channel
            self.att_c = keras.Sequential([   # (bs,w,h,2*out_ch) -> (bs,w,h,out_ch)
                layers.Conv2D(out_ch//g, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, kernel_initializer=init, use_biasuse_bias=use_bias),
                layers.Conv2D(out_ch, kernel_size=1, strides=1, activation=tf.sigmoid, kernel_initializer=init, use_bias=use_bias),
            ])
            
        if spatial_att: # 提炼重组一下mapping(即在(w,h)维度)
            self.att_s = keras.Sequential([   # (bs,w,h,2) -> (bs,w,h,1)
                layers.Conv2D(1, kernel_size=7, strides=1, padding='SAME', activation=tf.sigmoid, kernel_initializer=init, use_bias=use_bias),
            ])
            
    def call(self, input_frames):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv_block(input_frames)
        
        if self.channel_att:
            # adaptive_avg_pool2d 将(w,h)大小变成任意大小，如下面变成了(1,1),则经过cat之后为 (bs,1,1,2*out_ch)
            spatial_size = fm.shape[1:3]
            fm_pool = tf.concat([tf.nn.max_pool(fm,spatial_size,1,'VALID'),
                                 tf.nn.max_pool(fm,spatial_size,1,'VALID')], axis=-1)
            att = self.att_c(fm_pool)  # (bs,1,1,out_ch)
            fm = fm * att              # (bs,w,h,out_ch)*(bs,1,1,out_ch) -> (bs,w,h,out_ch)      
        if self.spatial_att:
            # (bs,w,h,1) + (bs,w,h,1) -> (bs,w,h,2) channel上一个是mean，一个是max
            fm_pool = tf.concat([tf.math.reduce_mean(fm, axis=-1, keepdims=True), tf.math.reduce_max(fm, axis=-1, keepdims=True)], axis=-1)
            att = self.att_s(fm_pool) # (bs,w,h,1)
            fm = fm * att             # (bs,w,h,out_ch)*(bs,w,h,1) -> (bs,w,h,out_ch)
        return fm


# In[3]:


"""Noise Estimation"""
class FCN(Model):
    def __init__(self, color = False, channels = [16, 32, 64, 32, 16], 
                 channel_att=False, spatial_att=False, use_bias = True):
        super(FCN, self).__init__()
        
        self.color = 3 if color else 1
        self.channels = channels
        
        self.channels.append(self.color)
        self.conv_layers = []
        for c in self.channels:
            self.conv_layers.append(Basic(c, channel_att=channel_att, spatial_att=spatial_att, use_bias = use_bias))
    
    def call(self, inputs):
        outputs = inputs
        for layer in self.conv_layers:
            outputs = layer(outputs)
        return outputs

