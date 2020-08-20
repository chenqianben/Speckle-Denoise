#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models

# KPN基本网路单元 (bs,in_ch,w,h)-> (bs,out_ch,w,h)
# 如果是channel_att则按channel做了一下特征处理，如果是spatial_att则按空间（pixel）做了一下特征处理
class Basic(nn.Module): 
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
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


# In[19]:


class Unet(nn.Module):
    def __init__(self, color=False, blind_est=True, channel_att=False, spatial_att=False, core_bias=False):
        super(Unet, self).__init__()
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        self.color = 3 if color else 1
        in_channel = self.color * (1 if blind_est else 2)
        out_channel = self.color
        if core_bias:
            out_channel += self.color

        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 256, channel_att=False, spatial_att=False)

        self.conv5 = Basic(256+256, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv6 = Basic(256+128, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(128+64, 64, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = Basic(64, out_channel, channel_att=channel_att, spatial_att=spatial_att)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if blind estimation, it is same as data, Otherwise, it is the data concatenated with noise estimation map
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))

        conv5 = self.conv5(torch.cat([conv3, F.interpolate(conv4, scale_factor=2, mode='bilinear')], dim=1))
        conv6 = self.conv6(torch.cat([conv2, F.interpolate(conv5, scale_factor=2, mode='bilinear')], dim=1))
        conv7 = self.conv7(torch.cat([conv1, F.interpolate(conv6, scale_factor=2, mode='bilinear')], dim=1))

        core = self.outc(F.interpolate(conv7, scale_factor=2, mode='bilinear'))   # (bs, ch, h, w)
        
        output = core[:,:self.color,:,:]
        bias = None if not self.core_bias else core[:,self.color:,:,:]
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            output += bias
        
        return output


