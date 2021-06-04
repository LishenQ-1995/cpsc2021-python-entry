# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:50:10 2021

@author: Lishen Qiu
"""

#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
# from utils import qrs_detect, comp_cosEn, save_dict
# import matplotlib # 注意这个也要import一次
# import matplotlib.pyplot as plt
# from score_2021 import RefInfo   
import glob
from tqdm import tqdm
# import numpy as np
# import os
# import sys
import json
# import wfdb
# import pandas as pd
# from scipy.io import loadmat,savemat
# import random
import torch
import resunet_CPSC2021
# from loss_CPSC2021 import dice_loss,dice_coef
# from torch.utils.data import Dataset, DataLoader
# import time
"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
        
def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams

def challenge_entry(sample_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = resunet_CPSC2021.resunet().to(device)
    model.load_state_dict(torch.load('./CPSC2021_0.99398.pth'))#读模型
    
    target_path=sample_path
    data_interval = 20
    data_length = 1600
    # flag=1
    data_files = glob.glob(target_path+'*.dat')
    data_files=sorted(data_files)
    
    for file_name in tqdm(data_files):
        file_name=file_name[0:-4]
        index_=file_name.index('d')
        name_=file_name[index_:]
        [sig, length, fs]=load_data(file_name)
        label_signal=np.zeros((length,))#这个label长度与信号长度一样，AF标记维1，其他地方标记为0
    #    print("ll",length)
        # length2=length
        if length<data_length:
            sig_temp=np.zeros((data_length,2))
            sig_temp[0:length,:]=sig
            sig=sig_temp
            length=data_length
        cut_num =int((len(sig)-data_length)/data_interval+1)
        batch_size=64
        batch_data=np.zeros((batch_size,data_length,2))
        start_temp=np.zeros((batch_size,1))
        n=0
        # pre_AFPosition_start=[]
        # pre_AFPosition_end=[]
        
        label_start=[]#记录label_signal的起止点
        label_end=[]
        for k in range(cut_num+1):
            
            
            start=data_interval*k
            end_=start+data_length
            if start+data_length>length:
                start=length-data_length
                end_=length
            label_start.append(start)
            label_end.append(end_)
    #        print(start,end_)
            temp=sig[start:end_,:]
            batch_data[n,:,:]=temp
            start_temp[n]=start
            n=n+1
            if n==batch_size or start+data_length>length:
                batch_data2=np.expand_dims(batch_data,axis=2)
                batch_data2=torch.Tensor(batch_data2)
                batch_data2=batch_data2.permute(0,3,1,2)
                batch_data2=batch_data2.to(device)
                
                pre_label1=model(batch_data2)
                pre_label1=pre_label1.data.cpu().numpy()
                pre_label1[pre_label1>=0.5]=1
                pre_label1[pre_label1<0.5]=0#64,1
                
            
                
                
                for j in range(n):
                    if pre_label1[j,0,0]==1:
                        label_signal[label_start[j]:label_end[j],]=1
                
                n=0
                label_start=[]#记录label_signal的起止点
                label_end=[] 
    
        label_signal[0,]=0
        label_signal[-1,]=0   
               
        pre_label_diff=np.diff(label_signal)
    
        AF_start_batch=np.where(pre_label_diff==1)[0]
        AF_end_batch=np.where(pre_label_diff==-1)[0]
        
        valid_index=[]
        for m in range(len(AF_start_batch)):
            if AF_end_batch[m]-AF_start_batch[m]>=385 :
                valid_index.extend([int(m)])
        AF_start_batch=AF_start_batch[valid_index]
        AF_end_batch=AF_end_batch[valid_index]
        
        
        AF_start_batch=np.array(sorted(AF_start_batch.copy()),dtype="float64")
        AF_end_batch=np.array(sorted(AF_end_batch.copy()),dtype="float64")
        pre_position=[]   


        if len(AF_start_batch)>0:
            if np.sum(label_signal)/len(label_signal)>=0.90 or len(AF_start_batch)>=30:
                pre_position.append([0,len(label_signal)-1]) 
            elif np.sum(label_signal)/len(label_signal)<0.10:
                pre_position=[]
            
            else:
                
                for m in range(len(AF_start_batch)):
                    if (AF_end_batch[m]-800)-(AF_start_batch[m]+800)>=385:

                        pre_position.append([AF_start_batch[m]+800,AF_end_batch[m]-800])
        else:
            pre_position=[]
        pred_dcit = {'predict_endpoints': pre_position}
        # save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
        save_dict(os.path.join(RESULT_PATH, name_+'.json'), pred_dcit)
    return pred_dcit


if __name__ == '__main__':


    # DATA_PATH="/home/ps/cwq/2021生理参数挑战赛/代码4房颤分类/分数/验证集数据/验证集数据/"
    # RESULT_PATH="/home/ps/cwq/2021生理参数挑战赛/代码4房颤分类/分数/验证集数据/验证集结果/"
    
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    # test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    # for i, sample in enumerate(test_set):
        # print(sample)
    # sample_path = os.path.join(DATA_PATH, sample)
    pred_dict = challenge_entry(DATA_PATH)

        

