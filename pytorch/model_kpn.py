#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models


# In[8]:


# KPN基本网路单元 (bs,in_ch,w,h)-> (bs,out_ch,w,h)
# 如果是channel_att则按channel做了一下特征处理，如果是spatial_att则按空间（pixel）做了一下特征处理
class Basic(nn.Module): 
    def __init__(self, in_ch, out_ch, g=4, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(   # (bs,in_ch,w,h) -> (bs,out_ch,w,h)
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att: # 提炼重组一下各个channel
            self.att_c = nn.Sequential(  # (bs,2*out_ch,w,h) -> (bs,out_ch,w,h)
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att: # 提炼重组一下mapping(即在(w,h)维度)
            self.att_s = nn.Sequential( # (bs,2,w,h) -> (bs,1,w,h)
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)   # (bs,out_ch,w,h)
        if self.channel_att:
            # adaptive_avg_pool2d 将(w,h)大小变成任意大小，如下面变成了(1,1),则经过cat之后为 (bs,2*out_ch,1,1)
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)   # (bs,out_ch,1,1)
            fm = fm * att               # (bs,out_ch,w,h)*(bs,out_ch,1,1) -> (bs,out_ch,w,h)
        if self.spatial_att:
            # (bs,1,w,h) + (bs,1,w,h) -> (bs,2,w,h) channel上一个是mean，一个是max
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool) # (bs,1,w,h)
            fm = fm * att             # (bs,out_ch,w,h)*(bs,1,w,h) -> (bs,out_ch,w,h)
        return fm


class KPN(nn.Module):
    def __init__(self, color=False, burst_length=8, blind_est=False, kernel_size=[3], sep_conv=False, 
                 channel_att=False, spatial_att=False, core_bias=False):
        # 注意，输入参数 kerel_size是跟kernel prediction networks一样的filter size
        # 注意，如果是blind_est，则输入端不加入noise 的先验知识，否则要将一个input channel留给noise
        # 注意，burst_length是时间序列，如video放映那种
        # 注意，sep_conv见156行解释
        super(KPN, self).__init__()
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length

        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 256, channel_att=False, spatial_att=False)

        self.conv5 = Basic(256+256, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv6 = Basic(256+128, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(128+64, 64, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = Basic(64, out_channel, channel_att=channel_att, spatial_att=spatial_att)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights) # 调用函数初始化

    @staticmethod  # 层层调用函数，相当于(staticmethod(_init_weights(m)))
    def _init_weights(m): # 初始化函数
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if blind estimation, it is same as data, Otherwise, it is the data concatenated with noise estimation map
        :param data:
        :return: pred_img_i and img_pred
        """
        (bs, N, c, h, w) = data_with_est.shape
        data_with_est = data_with_est.view(bs, -1, h, w)
        
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))

        conv5 = self.conv5(torch.cat([conv3, F.interpolate(conv4, scale_factor=2, mode='bilinear')], dim=1))
        conv6 = self.conv6(torch.cat([conv2, F.interpolate(conv5, scale_factor=2, mode='bilinear')], dim=1))
        conv7 = self.conv7(torch.cat([conv1, F.interpolate(conv6, scale_factor=2, mode='bilinear')], dim=1))

        core = self.outc(conv7)
        
        pred_img, pred_img_i = self.kernel_pred(data, core, white_level)
        
        return pred_img, pred_img_i                               # data是原数据, core是经过了KPN的noise数据
                                                         # data (bs,in_ch,w,h), core (bs,out_ch,w,h)
                                                         # 返回的是 pred_img_i: (bs, N, color, w, h)
                                                         # 和 pred_img: (bs, color, w, h)


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[3], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width): 
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width 或者 batch*(N*K^2)*height*width
        :core(bs,out_ch,w,h), out_ch是KPN的,即N*K^2,K是filter大小,N是时序 ####
        :return:
        """
        kernel_total = sum(self.kernel_size)  # 这里为5
        core = core.view(batch_size, N, -1, color, height, width)  # (bs,N,2*K,color,w,h)
        
        # 这里core_1是前K个，core_2是core_1之后K个，core_3是bias那一个
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)  
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            # einsum爱因斯坦求和，这里表示第三和第四维度相乘，kl*lm -> (km)
            # 表示：filters的最终数量K*K是可以看作由两个kernel vector K*1相乘得来的，这是矩阵因式分解，由sep_conv参数决定要不要这么做
            # 如果sep_conv，大大减少计算量，可以这么做是因为考虑了一个K*K的filter有行/列相似特征..?
            # 如果不sep_conv,计算量复杂很多，但是模型也更强大
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()  # core_out[K]:(bs,N,K*K,color,w,h)

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)           # (bs, N, K*K, color, w, h)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0): 
        """
        compute the pred image according to core and frames
        :param frames: (bs,N,h,w)(gray) or (bs,N,h,w,color)(color)
        :param core: (bs,out_ch,h,w)，out_ch是KPN的,即N*K*K(+1)
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size() # N这里应该是FPN的时序
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        
        if self.sep_conv: # core -> (kernel, bs, N, K * K, color, w, h)
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:  # img_stack为空列表时
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])  # 只有四个输入，则是对最后两个维度的左右上下依次做pad
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[..., i:i + height, j:j + width])   # 元素:(bs,N,color,h,w)
                img_stack = torch.stack(img_stack, dim=2)                             # img_stack:(bs,N,K*K,color,h,w)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            #print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(                          # core[K]:(bs,N,K*K,color,w,h), pred_img: (kernel,bs,N,color,w,h)
                core[K].mul(img_stack), dim=2, keepdim=False              # 元素:(bs, N, color, w, h)
            ))
        pred_img = torch.stack(pred_img, dim=0)                           # pred_img: (kernel, bs, N, color, w, h)
        #print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)           # pred_img_i: (bs, N, color, w, h), 对kernel做了平均
        
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        #print('white_level', white_level)
        pred_img_i = pred_img_i / white_level
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)           # pred_img: (bs, color, w, h)，对时序（帧数）做了平均
        #print('pred_img:', pred_img.size())
        #print('pred_img_i:', pred_img_i.size())
        return pred_img, pred_img_i


# In[9]:


class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) +                self.l1_loss(self.gradient(pred), self.gradient(ground_truth))


class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )

class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)




