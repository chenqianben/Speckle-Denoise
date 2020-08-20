#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers


# # global DFN 

# In[2]:


"""Defining the block"""
class Basic(Model):  
    def __init__(self, out_ch, g=16, channel_att=False, spatial_att=False,
                 init = tf.keras.initializers.he_normal(), bias_init = tf.zeros_initializer()):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.conv = keras.Sequential([
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
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


# In[3]:


"""Defining the model"""
class GDFN(Model): 
    def __init__(self, color = False, num_filters = 5, channel_att=False, spatial_att=False):
        super(GDFN, self).__init__()
        
        self.color = 3 if color else 1
        self.num_filters = num_filters
        out_channel = self.color * self.color * self.num_filters
        
        # encoder，注意，maxpool和upsampling是没有模型参数的，也可以直接写在__call__里面当函数调用
        self.conv1 = Basic(64, channel_att=False, spatial_att=False)  
        self.conv2 = Basic(128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, channel_att=False, spatial_att=False)
        self.conv5 = Basic(256, channel_att=False, spatial_att=False)
        
        self.avgpool1 = layers.AveragePooling2D(pool_size=(2, 2))
        self.avgpool2 = layers.AveragePooling2D(pool_size=(2, 2))
        self.avgpool3 = layers.AveragePooling2D(pool_size=(2, 2))
        self.avgpool4 = layers.AveragePooling2D(pool_size=(2, 2))
        
        self.avgpool1d = layers.AveragePooling2D(pool_size=(16, 16))
        self.avgpool2d = layers.AveragePooling2D(pool_size=(8, 8))
        self.avgpool3d = layers.AveragePooling2D(pool_size=(4, 4))
        self.avgpool4d = layers.AveragePooling2D(pool_size=(2, 2))

        self.conv6 = Basic(128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(64, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.avgpoolout = layers.AveragePooling2D(pool_size=(4, 4), strides=(1, 1))
        
            
    def call(self, ims):        # ims (bs, h, w, color)
        # encoder: (bs, w, h, in_channel) -> (bs, w/16, h/16, 512)
        conv1 = self.conv1(ims)
        conv2 = self.conv2(self.avgpool1(conv1))
        conv3 = self.conv3(self.avgpool2(conv2))
        conv4 = self.conv4(self.avgpool3(conv3))
        conv5 = self.conv5(self.avgpool4(conv4))
        
        conv1d = self.avgpool1d(conv1)
        conv2d = self.avgpool2d(conv2)
        conv3d = self.avgpool3d(conv3)
        conv4d = self.avgpool4d(conv4)
        
        conv6 = self.conv6(tf.concat([conv1d,conv2d,conv3d,conv4d,conv5], axis=-1))
        conv7 = self.conv7(conv6)
        core = self.conv8(conv7)  
        core = self.avgpoolout(core)    # (bs, kh, kw, c)
        
        bs, kh, kw, _ = core.shape
        
        outputs = []
        for n, cur_im in enumerate(ims):
            cur_im = tf.expand_dims(cur_im, axis=0)
            box = []
            for i in range(self.num_filters):
                cur_core = tf.reshape(core[n, :, :, i*self.color**2:(i+1)*self.color**2], 
                                      [kh, kw, self.color, self.color])                                  # (kh, kw, color, color)
                box.append(tf.nn.conv2d(cur_im, cur_core, strides=[1,1,1,1], padding='SAME'))            # (bs, h, w, color)
            cur_im = tf.reduce_mean(tf.stack(box, axis=0), axis=0, keepdims=False)
            outputs.append(cur_im)
        outputs = tf.concat(outputs, axis=0)
        return outputs, core




