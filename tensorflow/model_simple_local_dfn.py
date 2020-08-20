#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers


# Basic Block
"""Defining the model (using a Sonnet module)"""
class Basic_baseline(Model):  # snt model(version 2.0)有一个__init__，一个__call__(相当于call)
    def __init__(self, out_ch, if_normalized = False,
                 init = tf.keras.initializers.glorot_normal(), bias_init = tf.zeros_initializer(),):
        super(Basic_baseline, self).__init__()
        
        if if_normalized:
            self.conv = keras.Sequential([
                layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
            ])
        else:
            self.conv = keras.Sequential([
                layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
                layers.LeakyReLU(),
            ])
            
    def call(self, input_frames):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv(input_frames)   # (bs,w,h,out_ch)
        return fm

    
    
"""Defining the model (using a Sonnet module)"""
class Basic(Model):  # snt model(version 2.0)有一个__init__，一个__call__(相当于call)
    def __init__(self, out_ch, g=16, init = tf.keras.initializers.he_normal(), bias_init = tf.zeros_initializer(),
                 channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.conv = keras.Sequential([
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init),
            #layers.BatchNormalization(axis=-1),
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
        fm = self.conv(input_frames)   # (bs,w,h,out_ch)
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


# DnCNN
class DnCNN(Model):
    def __init__(self, color = False, inter_channels = 64):
        super(DnCNN, self).__init__()
        
        self.color = 3 if color else 1
        out_channel = (3 if color else 1)
        
        init = tf.keras.initializers.he_normal()
        bias_init = tf.zeros_initializer()
        
        self.conv1 = layers.Conv2D(inter_channels, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init)
        
        self.conv_layers = []
        for _ in range(18):
            self.conv_layers.append(Basic_baseline(inter_channels, if_normalized=True))
        
        self.conv_out = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding='SAME', kernel_initializer=init, bias_initializer=bias_init)
        
    def call(self, inputs):
        outputs = self.conv1(inputs)
        for layer in self.conv_layers:
            outputs = layer(outputs)
        outputs = self.conv_out(outputs)
        return inputs - outputs



# CBDNet
class FCN(Model):
    def __init__(self, color = False):
        super(FCN, self).__init__()
        
        self.color = 3 if color else 1
        self.conv_layers = []
        for _ in range(5):
            self.conv_layers.append(Basic_baseline(self.color, if_normalized=False))
    
    def call(self, inputs):
        outputs = inputs
        for layer in self.conv_layers:
            outputs = layer(outputs)
        return outputs

class CBDNet(Model):
    def __init__(self, color = False):
        super(CBDNet, self).__init__()
        
        self.noise_est = FCN(color)
        self.color = 3 if color else 1
        out_channel = 3 if color else 1
        
        # encoder，注意，maxpool和upsampling是没有模型参数的，也可以直接写在__call__里面当函数调用
        self.conv1 = Basic(128)  
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Basic(256)
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Basic(512)

        
        # decoder
        self.up4 = layers.Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2), padding='SAME')
        self.conv4 = Basic(256)
        self.up5 = layers.Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding='SAME')
        self.conv5 = Basic(128)
        self.outc = layers.Conv2D(out_channel, kernel_size=1, strides=1, padding='VALID')

    def call(self, inputs):        # input_frame (bs, h, w, color)

        ###############################
        #  filter-generating network  #
        ###############################
        # noise estimation
        noise_level = self.noise_est(inputs)         # (bs, h, w, color)
        
        inputs_noise = tf.concat([inputs, noise_level], axis=-1)
        
        # encoder: (bs, w, h, in_channel) -> (bs, w/16, h/16, 512)
        conv1 = self.conv1(inputs_noise)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        
        #  decoder (bs , w/16, h/16, 512) -> (bs, w, h, out_channel)     
        conv4 = self.conv4(conv2 + self.up4(conv3))
        conv5 = self.conv5(conv1 + self.up5(conv4))
        
        dynamic_filters = self.outc(conv5)            # (bs, h, w, color)
        
        #########################
        #  transformer network  #
        #########################     
        
        output_dynconv = inputs - dynamic_filters
        
        return output_dynconv, noise_level


# Loss
class CBDLoss(Model):
    """CBDNet loss"""
    def __init__(self, l1 = False, alpha = 0.5, beta = 0.05):
        super(CBDLoss, self).__init__()
        self.loss = keras.losses.MeanAbsoluteError() if l1 else keras.losses.MeanSquaredError()
        self.alpha = alpha
        self.beta = beta
        
    def call(self, pred, ground_truth, est_noise, gt_noise):
        return self.loss(pred, ground_truth) +                 self.alpha * tf.reduce_mean(tf.multiply(tf.abs(0.3 - tf.nn.relu(gt_noise - est_noise)), tf.square(est_noise - gt_noise))) +                 self.beta * tf.reduce_mean(tf.square(tf.image.image_gradients(est_noise)))

