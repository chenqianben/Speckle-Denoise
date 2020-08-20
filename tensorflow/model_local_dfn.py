#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers


# # Local DFN 

# In[2]:


"""Defining the block"""
class Basic(Model):  
    def __init__(self, out_ch, g=4, channel_att=False, spatial_att=False,
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


# In[10]:


"""Defining the model"""
class DFN(Model): 
    def __init__(self, color = False, filter_size=(3,3), channel_att=False, spatial_att=False):
        super(DFN, self).__init__()
        
        self.color = 3 if color else 1
        self.filter_size = filter_size
        out_channel = (3 if color else 1) * (filter_size[0]*filter_size[1])
        
        # encoder，注意，maxpool和upsampling是没有模型参数的，也可以直接写在__call__里面当函数调用
        self.conv1 = Basic(64, channel_att=False, spatial_att=False)  
        self.avgpool1 = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv2 = Basic(128, channel_att=False, spatial_att=False)
        self.avgpool2 = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv3 = Basic(256, channel_att=False, spatial_att=False)
        self.avgpool3 = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv4 = Basic(256, channel_att=False, spatial_att=False)

        # decoder
        self.up4 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.conv5 = Basic(256, channel_att=channel_att, spatial_att=spatial_att)
        self.up5 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.conv6 = Basic(128, channel_att=channel_att, spatial_att=spatial_att)
        self.up6 = layers.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.conv7 = Basic(64, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = Basic(out_channel, channel_att=channel_att, spatial_att=spatial_att)
            
    def call(self, ims):        # ims (bs, h, w, color)

        ###############################
        #  filter-generating network  #
        ###############################
        # encoder: (bs, w, h, in_channel) -> (bs, w/16, h/16, 512)
        conv1 = self.conv1(ims)
        conv2 = self.conv2(self.avgpool1(conv1))
        conv3 = self.conv3(self.avgpool2(conv2))
        conv4 = self.conv4(self.avgpool3(conv3))
        
        #  decoder (bs , w/16, h/16, 512) -> (bs, w, h, out_channel)
        conv5 = self.conv5(tf.concat([conv3, self.up4(conv4)], axis=-1))
        conv6 = self.conv6(tf.concat([conv2, self.up5(conv5)], axis=-1))
        conv7 = self.conv7(tf.concat([conv1, self.up6(conv6)], axis=-1))
        dynamic_filters = self.outc(conv7)    # (bs, h, w, color*filter_size*filter_size)
        
        bs, h, w, _ = dynamic_filters.shape
        dynamic_filters = tf.reshape(dynamic_filters, [bs, h, w, self.color, -1]) 
        #dynamic_filters = tf.nn.softmax(dynamic_filters, axis=-1)
        
        #########################
        #  transformer network  #
        #########################
        assert self.filter_size[0] == self.filter_size[1]
        bs, h, w, c = ims.shape
        K = self.filter_size[0]

        ims_pad = tf.pad(ims, paddings=[[0,0], [K//2,K//2], [K//2,K//2], [0,0]], mode='constant')
        ims_transformed = []
        
        for i in range(K):
            for j in range(K):
                ims_transformed.append(ims_pad[:,i:i+h, j:j+w,:])
        ims_transformed = tf.stack(ims_transformed, axis=-1)     # (bs h, w, c, K*K)
        output_dynconv = tf.reduce_mean(dynamic_filters * ims_transformed, axis=-1, keepdims=False) # (bs, h, w, color)
        
        return output_dynconv, dynamic_filters


# ## Test

# ### Baseline

# In[11]:


if __name__ == '__main__':
    model = DFN(color = False, filter_size=(3,3), channel_att=False, spatial_att=False)

    data = tf.cast(tf.convert_to_tensor(np.arange(128*128*1).reshape(1,128,128,1)), tf.float32)
    output, _ = model(data)
    print(output.shape)
    print(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))


# ## loss func

# In[5]:


class TensorGradient(Model):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def call(self, img):  # (bs, h, w, c)
        h, w = img.shape[1], img.shape[2]
        l = tf.pad(img, paddings=[[0,0], [0,0], [1,0], [0,0]], mode='constant')  # paddings order: [dim 1 [before, after], dim2...]
        r = tf.pad(img, paddings=[[0,0], [0,0], [0,1], [0,0]], mode='constant')
        u = tf.pad(img, paddings=[[0,0], [1,0], [0,0], [0,0]], mode='constant')
        d = tf.pad(img, paddings=[[0,0], [0,1], [0,0], [0,0]], mode='constant')
        if self.L1:
            return K.abs((l - r)[:, 0:w, 0:h, :]) + K.abs((u - d)[:, 0:w, 0:h, :])
        else:
            return K.sqrt(
                K.pow((l - r)[:, 0:w, 0:h, :], 2) + K.pow((u - d)[:, 0:w, 0:h, :], 2)
            )
        
class CombinedLoss(Model):
    """
    Combined loss function.
    """
    def __init__(self, gradient_L1=True):
        super(CombinedLoss, self).__init__()
        self.l1_loss = keras.losses.MeanAbsoluteError()
        self.l2_loss = keras.losses.MeanSquaredError()
        self.gradient = TensorGradient(gradient_L1)

    def call(self, pred, ground_truth):
        
        return self.l2_loss(pred, ground_truth) +                self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


# In[6]:


class DFNLoss(Model):
    """
    DFN loss function.
    """
    def __init__(self, filter_size, alpha, color):
        super(DFNLoss, self).__init__()
        self.l1_loss1 = keras.losses.MeanAbsoluteError()
        self.l1_loss2 = keras.losses.MeanAbsoluteError()
        self.filter_size = filter_size
        self.alpha = alpha
        self.color = 3 if color else 1
        
    def call(self, pred, ground_truth, anneal_ground_truth, anneal_noise, dynamic_filters):
        bs, h, w, color = ground_truth.shape
        anneal_ground_truth = tf.reshape(anneal_ground_truth, [-1, self.filter_size[0]*self.filter_size[1], h, w, color])
        anneal_ims_noise = tf.reshape(anneal_ims_noise, [-1, self.filter_size[0]*self.filter_size[1], h, w, color])
        
        anneal_pred = anneal_noise - dynamic_filters
        return self.l1_loss1(ground_truth, pred) + self.alpha * self.l1_loss2(anneal_ground_truth, anneal_pred)


# ## test

# In[7]:


if __name__ == '__main__':
    data = tf.cast(tf.convert_to_tensor(np.arange(512*512*3).reshape(1,512,512,3)), tf.float32)

    loss_func = CombinedLoss(gradient_L1 = True)
    loss = loss_func(data, data)
    print(loss)


# In[ ]:




