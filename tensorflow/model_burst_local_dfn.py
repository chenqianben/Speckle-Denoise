#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers


# # model

# In[2]:


"""Defining the block"""
class Basic(Model): 
    def __init__(self, out_ch, g=4, channel_att=False, spatial_att=False,
                 init = tf.keras.initializers.he_normal(), bias_init = tf.zeros_initializer()):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.conv_block = keras.Sequential([
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


"""Defining the block"""
class KPN(Model):
    def __init__(self, color=False, burst_length=8, blind_est=False, kernel_size=[5], 
                 channel_att=False, spatial_att=False, core_bias=False):
        # 注意，输入参数 kerel_size是跟kernel prediction networks一样的filter size
        # 注意，如果是blind_est，则输入端不加入noise 的先验知识，否则要将一个input channel留给noise
        # 注意，burst_length是时间序列，如video放映那种
        # 注意，sep_conv参数还没有加进去..............
        super(KPN, self).__init__()
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (burst_length * np.sum(np.array(kernel_size) ** 2))
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        
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
        
        self.kernel_pred = KernelConv(kernel_size, self.core_bias)
        
    def call(self, data_with_est, data): 
        """
        forward and obtain pred image directly
        :param data_with_est: 如果blind_est=False, 这应该是有noise作为一个单独channel加入的数据, 如果blind_est= True, 这就是frames (bs, N, h, w, c(+1))
        :param data: frames, 是一个seq组成的train_X  (bs, N, h, w, c)
        :return: pred_img_i and img_pred
        """
        
        # encoder: (bs, w, h, in_channel) -> (bs, w/16, h/16, 512)
        (bs, N, h, w, c) = data_with_est.shape
        data_with_est = tf.reshape(data_with_est, [bs, h, w, -1])
        
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(self.avgpool1(conv1))
        conv3 = self.conv3(self.avgpool2(conv2))
        conv4 = self.conv4(self.avgpool3(conv3))
        
        #  decoder (bs , w/16, h/16, 512) -> (bs, w, h, out_channel)
        conv5 = self.conv5(tf.concat([conv3, self.up4(conv4)], axis=-1))
        conv6 = self.conv6(tf.concat([conv2, self.up5(conv5)], axis=-1))
        conv6 = self.conv7(tf.concat([conv1, self.up6(conv6)], axis=-1))
        core = self.outc(conv6)    # (bs, h, w, color*filter_size*filter_size)
        
        pred_img, pred_img_i  = self.kernel_pred(data, core) # data是原始data, core是data_with_est由unet得到的dynamic filter
        return pred_img, pred_img_i                          # pred_img_i: (bs, N, w, h, color) 和 pred_img: (bs, w, h, color)


# In[12]:


# 使用列表创建frame_transformed
class KernelConv(Model):
    """
    the class of computing prediction by applying dynamic filter
    """
    def __init__(self, kernel_size=[5], core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.core_bias = core_bias

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
                k_diff = (kernel[index - 1]**2 - kernel[index]**2) // 2
                img_stack = img_stack[..., k_diff:-k_diff]
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

# In[13]:


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
    anneal loss function
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
    loss function of KPN
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


# # test

# ## extract_patches

# In[14]:


if __name__ == '__main__':
    data = tf.cast(tf.convert_to_tensor(np.arange(512*512*3).reshape(1,512,512,3)), tf.float32)
    a = tf.image.extract_patches(images=data, sizes=[1, 5, 5, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    print(a.shape)


# ## KernelConv

# In[15]:


if __name__ == '__main__':
    burst_length=8
    color = False
    blind_est = False

    sub_model = KernelConv(kernel_size=[3], core_bias=False)
    frames = tf.cast(tf.convert_to_tensor(np.arange(8*512*512*3).reshape(1,8,512,512,3)), tf.float32)
    core = tf.cast(tf.convert_to_tensor(np.arange(512*512*600).reshape(1,512,512,600)), tf.float32)  # 600 = (5*5)*8*3
    
    outputs, outputs_n = sub_model(frames, core)
    print(outputs.shape)
    print(outputs_n.shape)


# ## KPN

# In[16]:


import os           # 把graphviz放入到环境变量中，才能运行keras.utils.plot_model，显示图片
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

if __name__ == '__main__':
    burst_length=1
    color = False
    blind_est = True
    train_X_with_est = tf.cast(tf.convert_to_tensor(np.arange(128*128*1).reshape(1,1,128,128,1)), tf.float32) 
    train_X = tf.cast(tf.convert_to_tensor(np.arange(128*128*1).reshape(1,1,128,128,1)), tf.float32)          
    train_Y = tf.cast(tf.convert_to_tensor(np.arange(128*128*1).reshape(1,128,128,1)), tf.float32)   
    
    model = KPN(color=color, burst_length=burst_length, blind_est=blind_est, kernel_size=[3], 
                channel_att=False, spatial_att=False, core_bias=False)
    basic_loss_func = LossBasic()
    
    output, output_i = model(train_X_with_est, train_X)
    loss = basic_loss_func(output, train_Y)
    
    print(output.shape, output_i.shape)
    print(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))
    
    # save model as figure
    # tf.keras.utils.plot_model(model, to_file='model.png')


# ## loss

# In[10]:


if __name__ == '__main__':
    pred_img_i = tf.cast(tf.convert_to_tensor(np.arange(512*512*3).reshape(1,1,512,512,3)), tf.float32)
    pred_img = tf.cast(tf.convert_to_tensor(np.arange(512*512*3).reshape(1,512,512,3)), tf.float32)
    ground_truth = tf.cast(tf.convert_to_tensor(np.arange(512*512*3).reshape(1,512,512,3)), tf.float32)
    global_step=1

    loss_func = LossFunc(coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100)
    loss = loss_func(pred_img_i, pred_img, ground_truth, global_step)
    print(loss)


# In[ ]:




