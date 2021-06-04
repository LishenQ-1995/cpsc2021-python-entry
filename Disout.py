# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:25:53 2020

@author: Lishen Qiu
"""
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
"""
Beyond Dropout: Feature Map Distortion to Regularize Deep Neural Networks
https://arxiv.org/abs/2002.11022
Args:
    dist_prob (float): probability of an element to be distorted.
    block_size (int): size of the block to be distorted.
    alpha: the intensity of distortion.
Shape:
    - Input: `(N, C, H, W)`
    - Output: `(N, C, H, W)`
"""
class Disout(nn.Module):
    def __init__(self, dist_prob, block_size=6,alpha=1.0):
        super(Disout, self).__init__()

        self.dist_prob = dist_prob      
        self.weight_behind=None
  
        self.alpha=alpha
        self.block_size = block_size
        
    def forward(self, x):

        if not self.training: 
            return x
        else:
            x=x.clone()
            if x.dim()==4:           
                width=x.size(2)
                height=x.size(3)
                
   
                seed_drop_rate = self.dist_prob* (width*height) / self.block_size**2 / (( width -self.block_size + 1)*( height -self.block_size + 1))
                
                
                valid_block_center=torch.zeros(width,height,device=x.device).float()
                valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),int(self.block_size // 2):(height - (self.block_size - 1) // 2)]=1.0

                valid_block_center=valid_block_center.unsqueeze(0).unsqueeze(0)
               
                randdist = torch.rand(x.shape,device=x.device)
                
                block_pattern = ((1 -valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()
         
                if self.block_size == width and self.block_size == height:            
                    block_pattern = torch.min(block_pattern.view(x.size(0),x.size(1),x.size(2)*x.size(3)),dim=2)[0].unsqueeze(-1).unsqueeze(-1)
                else:
                    block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)

                if self.block_size % 2 == 0:
                        block_pattern = block_pattern[:, :, :-1, :-1]
                percent_ones = block_pattern.sum() / float(block_pattern.numel())


 
                if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                    wtsize=self.weight_behind.size(3)
                    weight_max=self.weight_behind.max(dim=0,keepdim=True)[0]
                    sig=torch.ones(weight_max.size(),device=weight_max.device)
                    sig[torch.rand(weight_max.size(),device=sig.device)<0.5]=-1
                    weight_max=weight_max*sig 
                    weight_mean=weight_max.mean(dim=(2,3),keepdim=True)
                    if wtsize==1:
                        weight_mean=0.1*weight_mean
                    #print(weight_mean)
                mean=torch.mean(x).clone().detach()
                var=torch.var(x).clone().detach()

                if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
                    dist=self.alpha*weight_mean*(var**0.5)*torch.randn(*x.shape,device=x.device)
                else:
                    dist=self.alpha*0.01*(var**0.5)*torch.randn(*x.shape,device=x.device)

            x=x*block_pattern
            dist=dist*(1-block_pattern)
            x=x+dist
            x=x/percent_ones
            return x
            
            
class LinearScheduler(nn.Module):
    def __init__(self, disout, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.disout = disout
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.disout(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.disout.dist_prob = self.drop_values[self.i]
        self.i += 1


class BasicBlock_disout(nn.Module):


    def __init__(self, in_plane, out_plane, stride=1, downsample=None,
                dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
       
        super(BasicBlock_disout, self).__init__()

        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.disout1=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.disout2=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)

    def forward(self, x):
       
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x=self.disout1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x=self.disout2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out