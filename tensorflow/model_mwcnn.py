#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers

from dwt import DWT, IDWT


"""Defining the block"""
class Basic(Model):  
    def __init__(self, out_ch, kernel_size = 3, g=16, channel_att=False, spatial_att=False, 
                 init = tf.keras.initializers.glorot_normal(), bias_init = tf.zeros_initializer()):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.conv = keras.Sequential([
            layers.Conv2D(out_ch, kernel_size=kernel_size, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=kernel_size, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=kernel_size, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])
            
        
        if channel_att: # 提炼重组一下各个channel
            self.att_c = keras.Sequential([   # (bs,w,h,2*out_ch) -> (bs,w,h,out_ch)
                layers.Conv2D(out_ch//g, kernel_size=1, strides=1, activation=tf.nn.leaky_relu, kernel_initializer=init, bias_initializer=bias_init),
                layers.Conv2D(out_ch, kernel_size=1, strides=1, activation=tf.sigmoid, kernel_initializer=init, bias_initializer=bias_init),
            ])
            
        if spatial_att: # 提炼重组一下mapping(即在(w,h)维度)
            self.att_s = keras.Sequential([   # (bs,w,h,2) -> (bs,w,h,1)
                layers.Conv2D(1, kernel_size=7, strides=1, padding='SAME', activation=tf.sigmoid, kernel_initializer=init, bias_initializer=bias_init),
            ])
            
    def call(self, input_frames):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv(input_frames)
        
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



"""Defining the model"""
class MWCNN(Model): 
    def __init__(self, color = False, kernel_size=(5,5), channel_att=False, spatial_att=False):
        super(MWCNN, self).__init__()
        
        self.color = 3 if color else 1
        self.kernel_size = kernel_size
        
        self.outc = self.color
        self.dwt_model = DWT()
        self.idwt_model = IDWT()
        
        # encoder，注意，maxpool和upsampling是没有模型参数的，也可以直接写在__call__里面当函数调用
        self.conv1 = Basic(64, kernel_size=self.kernel_size, channel_att=False, spatial_att=False)  
        self.conv2 = Basic(128, kernel_size=self.kernel_size, channel_att=False, spatial_att=False)
        self.conv3 = Basic(192, kernel_size=self.kernel_size, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, kernel_size=self.kernel_size, channel_att=False, spatial_att=False)

        # decoder
        self.conv5 = Basic(192, kernel_size=self.kernel_size, channel_att=channel_att, spatial_att=spatial_att)
        self.conv6 = Basic(128, kernel_size=self.kernel_size, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(64, kernel_size=self.kernel_size, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = Basic(self.outc, kernel_size=self.kernel_size, channel_att=channel_att, spatial_att=spatial_att)
            
    def call(self, ims):        # ims (bs, h, w, color)
        # encoder: (bs, h, w, color) -> (bs, w/16, h/16, 512)  
        conv1 = self.conv1(self.dwt_model(ims))    # (bs, h/2, w/2, color*4) -> (bs, h/2, w/2, 64)
        conv2 = self.conv2(self.dwt_model(conv1))  # (bs, h/4, w/4, 64*4) -> (bs, h/4, w/4, 128) 
        conv3 = self.conv3(self.dwt_model(conv2))  # (bs, h/8, w/8, 128*4) -> (bs, h/8, w/8, 192) 
        conv4 = self.conv4(self.dwt_model(conv3))  # (bs, h/16, w/16, 192*4) -> (bs, h/16, w/16, 256) 
        
        #  decoder (bs , w/16, h/16, 512) -> (bs, w, h, out_channel)
        conv5 = self.conv5(tf.concat([conv3, self.idwt_model(conv4)], axis=-1))
        conv6 = self.conv6(tf.concat([conv2, self.idwt_model(conv5)], axis=-1))
        conv7 = self.conv7(tf.concat([conv1, self.idwt_model(conv6)], axis=-1))
        outputs = self.outc(self.idwt_model(conv7))   
        
        return outputs



