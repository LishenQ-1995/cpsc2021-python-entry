# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:16:59 2020

@author: Lishen Qiu
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
#import keras.backend as K
#y_pred=
#y_true=
#
#Ncl = y_pred.shape[-1]
#w = np.zeros(shape=(Ncl,))
#w = np.sum(y_true, (0,2))
#w = 1/(w**2+0.000001)
## Compute gen dice coef:
#numerator = y_true*y_pred
#numerator = w*np.sum(numerator,(0,1,2))
#numerator = np.sum(numerator)
#denominator = y_true+y_pred
#denominator = w*np.sum(denominator,(0,1,2))
#denominator = np.sum(denominator)
#gen_dice_coef = 2*numerator/denominator
#return gen_dice_coef

#y_pred=torch.zeros(2,3,2000)
#y_true=torch.zeros(2,3,2000)
#y_true_f = torch.flatten(y_true)
#y_pred_f = torch.flatten(y_pred)
#
#print(y_pred.size())
#print(y_true_f.size())
#intersection = torch.sum(y_true_f * y_pred_f)

#a=[1,2,3,4,0,100]
#b=[5,6,7,1,9,1]
#top_line = np.minimum(a,b)


criterion1 = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = np.zeros(shape=(Ncl,))
    w = np.sum(y_true, axis=(0,1))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*np.sum(numerator,axis=(0,1,2))
    numerator = np.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*np.sum(denominator,axis=(0,1,2))
    denominator = np.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

def bceloss(y_pred,y_true):
    return nn.BCELoss(y_pred,y_true)
    
def dice_coef(y_pred,y_true,  smooth=1e-7):
#    print(111111,y_true.shape,y_pred.shape)
    
    
    
    

    dice_y_true=y_true[:,1:,:]
    dice_y_pred=y_pred[:,1:,:]
#    print(y_true[:,0,:],y_true[:,0,:]==1)
#    dice_y_true = dice_y_true[torch.squeeze(y_true[:,0,:]==1)] 
#    dice_y_pred = dice_y_pred[torch.squeeze(y_true[:,0,:]==1)] 
#    print('dice_y_pred',dice_y_pred.shape,dice_y_pred.shape)
    y_true_f = torch.flatten(dice_y_true)
    y_pred_f = torch.flatten(dice_y_pred)
    
    intersection = torch.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
def dice_loss(y_pred,y_true):
    class_y_true=y_true[:,0:1,:]
    class_y_pred=y_pred[:,0:1,:]
#    print(class_y_pred,class_y_true)
#    print('class_y_true',class_y_true.shape,class_y_true.shape)
    bceloss1 =criterion(class_y_pred,class_y_true)
    return bceloss1


def FocalLoss( inputs, targets):        
	# 计算正负样本权重
    
    Focall_y_true=targets
    Focall_y_pred=inputs
    
    y_true_f = torch.flatten(Focall_y_true)
    y_pred_f = torch.flatten(Focall_y_pred)
    
    alpha = 0.8
    gamma = 2
    logits = False
    reduce = True
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(y_pred_f, y_true_f, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(y_pred_f, y_true_f, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss
