#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model, layers

from dwt import DWT, IDWT


# # model

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


# In[73]:


class MWKPN(Model):
    def __init__(self, color=False, burst_length=8, blind_est=False, sep_conv=False, kernel_size=[5], 
                 channel_att=False, spatial_att=False, core_bias=False, use_bias=True):
        # 注意，输入参数 kerel_size是跟kernel prediction networks一样的filter size
        # 注意，如果是blind_est，则输入端不加入noise 的先验知识，否则要将一个input channel留给noise
        # 注意，burst_length是时间序列，如video放映那种
        super(MWKPN, self).__init__()
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.use_bias = use_bias
        self.color_channel = 3 if color else 1

        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        
        self.dwt_model = DWT()
        self.idwt_model = IDWT()
        
        # encoder，注意，maxpool和upsampling是没有模型参数的，也可以直接写在__call__里面当函数调用
        self.conv1 = Basic(64, channel_att=False, spatial_att=False)  
        self.conv2 = Basic(128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(192, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, channel_att=False, spatial_att=False)

        # decoder
        self.conv5 = Basic(192, channel_att=channel_att, spatial_att=spatial_att)
        self.conv6 = Basic(128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(64, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = Basic(out_channel, channel_att=channel_att, spatial_att=spatial_att)
        
        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
    def call(self, data_with_est, data): 
        """
        forward and obtain pred image directly

        :param data: frames, 是一个seq组成的train_X  (bs, N, h, w, c)
        :return: pred_img_i and img_pred
        """
        
        (bs, N, h, w, c) = data_with_est.shape
        data_with_est = tf.transpose(tf.reshape(data_with_est, [bs, -1, h, w]), [0, 2, 3, 1]) # attention! dim position must be correspondant when reshaped
   
        # encoder: (bs, h, w, color) -> (bs, w/16, h/16, 512)  
        conv1 = self.conv1(self.dwt_model(data_with_est))    # (bs, h/2, w/2, color*4) -> (bs, h/2, w/2, 64)
        conv2 = self.conv2(self.dwt_model(conv1))  # (bs, h/4, w/4, 64*4) -> (bs, h/4, w/4, 128) 
        conv3 = self.conv3(self.dwt_model(conv2))  # (bs, h/8, w/8, 128*4) -> (bs, h/8, w/8, 192) 
        conv4 = self.conv4(self.dwt_model(conv3))  # (bs, h/16, w/16, 192*4) -> (bs, h/16, w/16, 256) 
        
        #  decoder (bs , w/16, h/16, 512) -> (bs, w, h, out_channel)
        conv5 = self.conv5(tf.concat([conv3, self.idwt_model(conv4)], axis=-1))
        conv6 = self.conv6(tf.concat([conv2, self.idwt_model(conv5)], axis=-1))
        conv7 = self.conv7(tf.concat([conv1, self.idwt_model(conv6)], axis=-1))
        core = self.outc(self.idwt_model(conv7))    
        
        pred_img, pred_img_i  = self.kernel_pred(data, core) # data是原始data, core是data_with_est由unet得到的dynamic filter
        return pred_img, pred_img_i                     # pred_img_i: (bs, N, w, h, color) 和 pred_img: (bs, w, h, color)



# 使用列表创建frame_transformed
class KernelConv(Model):
    """
    the class of computing prediction by applying dynamic filter
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias
        
    def _sep_conv_core(self, core, batch_size, N, height, width, color): 
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*height*width*(N*2*K) 或者 batch*height*width*(N*K^2)
        :core(bs,w,h,out_ch), out_ch是KPN的,即N*K^2,K是filter大小,N是时序 ####
        :return:
        """
        kernel_total = sum(self.kernel_size)  
        core = tf.reshape(core, [batch_size, N, height, width, color, -1])  # (bs,N,w,h,color,2*K)
        
        # 这里core_1是前K个，core_2是core_1之后K个，core_3是bias那一个
        if not self.core_bias:
            core_1, core_2 = tf.split(core, kernel_total, axis=-1)
        else:
            core_1, core_2, core_3 = tf.split(core, [kernel_total, kernel_total, 1], axis=-1)  
            
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = tf.reshape(core_1[..., cur:cur + K], [batch_size, N, height, width, color, K, 1])
            t2 = tf.reshape(core_2[..., cur:cur + K], [batch_size, N, height, width, color, 1, K])
            # einsum爱因斯坦求和，这里表示第三和第四维度相乘，kl*lm -> (km)
            # 表示：filters的最终数量K*K是可以看作由两个kernel vector K*1相乘得来的，这是矩阵因式分解，由sep_conv参数决定要不要这么做
            # 如果sep_conv，大大减少计算量，可以这么做是因为考虑了一个K*K的filter有行/列相似特征..?
            # 如果不sep_conv,计算量复杂很多，但是模型也更强大
            core_out[K] = tf.reshape(tf.einsum('ijklmno,ijklmop->ijklmnp', t1, t2), [batch_size, N, height, width, color, K*K])
            cur += K
            #print(core_out[K].shape)
        # it is a dict
        return core_out, None if not self.core_bias else tf.squeeze(core_3, axis=-1)  # core_out[K]:(bs,N,K*K,w,h,color)

    def _convert_dict(self, core, batch_size, N, height, width, color):
        """
        separate the channel dimension to three parts: burst_length, kernel_size, bias if core_bias = True
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: bs * h * w * (N*K*K(+1))
        :return: core_out, a dict
        """
        core_out = {}
        core = tf.reshape(core, [batch_size, N, height, width, color, -1])  # (bs, N, w, h, color, K*K(+1))
        
        kernel = self.kernel_size[::-1]
        ind = 0
        for K in kernel:
            core_out[K] = core[..., ind:ind+K**2]
            ind += K**2
        bias = None if not self.core_bias else core[..., -1]
        return core_out, bias
    
    def call(self, frames, core):               
        """
        compute the pred image according to core and frames
        :param frames: (bs,N,h,w,1)(gray) or (bs,N,h,w,3)(color)
        :param core: (bs,h,w,out_ch)，out_ch是KPN的,即N*K*K(+1)
        :return:
        """
        assert len(frames.shape) == 5
        batch_size, N, height, width, color = frames.shape # N这里应该是FPN的时序
        
        if self.sep_conv: # core -> (kernel, bs, N, K * K, color, w, h)
            core, bias = self._sep_conv_core(core, batch_size, N, height, width, color)
        else:
            core, bias = self._convert_dict(core, batch_size, N, height, width, color) # core is a dict, core[K]:(bs, N, h, w, c, K*K(+1))
        
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not len(img_stack):
                frame_pad = tf.pad(frames, paddings=[[0,0], [0,0], [K//2,K//2], [K//2,K//2], [0,0]], mode='constant')
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[:, :, i:i+height, j:j+width,:])
                img_stack = tf.stack(img_stack, axis=-1)                 # (bs, N, h, w，color, K*K) 
            else:
                # k_diff = (kernel[index - 1]**2 - kernel[index]**2) // 2
                k_diff = (kernel[index-1] - kernel[index]) // 2
                k_chosen = []
                for i in range(k_diff, kernel[index-1]-k_diff):
                    k_chosen += [i*kernel[index-1]+j for j in range(k_diff, kernel[index-1]-k_diff)]
                # img_stack = img_stack[..., k_diff:-k_diff]
                img_stack = tf.convert_to_tensor(img_stack.numpy()[..., k_chosen])
            pred_img.append(tf.reduce_sum(tf.math.multiply(core[K], img_stack), axis=-1, keepdims=False))
        pred_img = tf.stack(pred_img, axis=0)                           # (nb_kernels, bs, N, h, w, color)
        pred_img_i = tf.reduce_mean(pred_img, axis=0, keepdims=False)   # (bs, N, h, w, color)
        
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
            
        pred_img = tf.reduce_mean(pred_img_i, axis=1, keepdims=False)          # (bs, h, w, color)

        return pred_img, pred_img_i


# # loss func

# In[75]:


class TensorGradient(Model):
    """
    Gradient of tensor
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
        
class LossBasic(Model):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = keras.losses.MeanAbsoluteError()
        self.l2_loss = keras.losses.MeanSquaredError()
        self.gradient = TensorGradient(gradient_L1)

    def call(self, pred, ground_truth):
        
        return self.l2_loss(pred, ground_truth) +                self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal(Model):
    """
    Anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def call(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, height, width, color]
        :param ground_truth: [batch_size, height, width, color]
        :return:
        """
        loss = 0
        for i in range(pred_i.shape[1]):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.shape[1]
        return self.beta * self.alpha**np.array(global_step,dtype=np.float32) * loss

class LossFunc(Model):
    """
    Loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def call(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, height, width, color]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, height, width, color]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)
