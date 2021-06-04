import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

from Disout import Disout,LinearScheduler
from math import floor, ceil
from math import sqrt
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
#        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class PSPModule(nn.Module):
    def __init__(self, features, out_features=256, sizes=(41, 9, 11, 5)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
#        print(self.stages)
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)

#        self.relu = nn.ReLU()
        self.relu = Mish()
#        self.bn=nn.BatchNorm2d(out_features)
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, 1))
       
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
#        bn=nn.BatchNorm2d(features)#bn是自己加的
#        relu_=nn.ReLU()
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
#        print(h, w)
#        print(feats[0].size())
#        print(self.stages)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
#        print("0",priors[0].size())
#        print("1",priors[1].size())
#        print("2",priors[2].size())
#        print("3",priors[3].size())
#        print("4",torch.cat(priors, 1).size())
        bottle = self.bottleneck(torch.cat(priors, 1))
#        print("5",bottle.size())
#        bottle=self.bn(bottle)
        return self.relu(bottle)


class LSTM_ATTENTION(nn.Module):
    '''
    input_size：x的特征维度，输入的维度，序列长度不要求输入
    hidden_size：隐藏层的特征维度,输出维度
    num_layers：lstm隐层的层数，默认为1，叠加的层数
    bias：False则bih=0和bhh=0. 默认为True
    batch_first：True则输入输出的数据格式为 (batch, seq, feature)
    dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
    bidirectional：True则为双向lstm默认为False
    输入：input, (h0, c0)
    输出：output, (hn,cn)
    '''
    def __init__(self,input_size,hidden_size,num_layers ):
        super(LSTM_ATTENTION,self).__init__()
        self.Bidirectional1=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers =num_layers ,bidirectional=False,batch_first=True)
    def forward(self,x):
        out=self.Bidirectional1(x)
        return out
        
        
        
        
        
        
        
        
        
class block_down(nn.Module):
    def __init__(self,in_channels, channels,kernel_size=(11,1),resnet=False):
        super(block_down,self).__init__()
        self.resnet=resnet
        self.conv1=nn.Conv2d (in_channels, channels, kernel_size=kernel_size,
                               stride=1, padding=(int((kernel_size[0]-1)/2),0), bias=False)  #s=1时，padding=(f-1)/2
        self.bn1=nn.BatchNorm2d(channels)
#        self.dropout1 = nn.Dropout(0.4)
        self.dropout1=Disout(dist_prob=0.2,block_size=6,alpha=30)

        self.conv2=nn.Conv2d (channels, channels, kernel_size=(kernel_size),
                               stride=1, padding=(int((kernel_size[0]-1)/2),0), bias=False)
        self.bn2=nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        self.relu = Mish()
        
    def forward(self,x):
        out=self.relu(self.bn1((self.conv1(x))))
        out=self.dropout1(out)
        out=self.conv2(out)
        if self.resnet==True:
            out=out+self.shortcut(x)
        out=self.relu(self.bn2(out))
        return out
class block_up(nn.Module):
    def __init__(self,in_channels, channels,kernel_size=(11,1),resnet=False):
        super(block_up,self).__init__()
        self.resnet=resnet
        
        
        self.conv1=nn.Conv2d (in_channels, channels, kernel_size=kernel_size,
                               stride=1, padding=(int((kernel_size[0]-1)/2),0), bias=False)  #s=1时，padding=(f-1)/2
        self.bn1=nn.BatchNorm2d(channels)
#        self.dropout1 = nn.Dropout(0.4)
        self.dropout1=Disout(dist_prob=0.2,block_size=6,alpha=30)
        self.conv2=nn.Conv2d (channels, channels, kernel_size=(kernel_size),
                               stride=1, padding=(int((kernel_size[0]-1)/2),0), bias=False)
        self.bn2=nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        self.relu = Mish()
    def forward(self,x):
        
        out=self.relu(self.bn1((self.conv1(x))))
        out=self.dropout1(out)
        out=self.conv2(out)
        if self.resnet==True:
            out=out+self.shortcut(x)
        out=self.relu(self.bn2(out))
        return out

class Depth_conv(nn.Module):
    def __init__(self,in_channels,kernel_size,shortcut):
        super(Depth_conv,self).__init__()
        self.shortcut=shortcut
        self.depth_conv1=nn.Conv2d(in_channels=in_channels,
                                    out_channels=20,
                                    kernel_size=(kernel_size,1),
                                    stride=1,
                                    padding=(int((kernel_size-1)/2),0),
                                    dilation=(1,1),
                                    groups=2)  
        self.bn1=nn.BatchNorm2d(20) 
        self.depth_conv2=nn.Conv2d(in_channels=20,
                                    out_channels=20,
                                    kernel_size=(kernel_size,1),
                                    stride=1,
                                    padding=(int((kernel_size-1)/2),0),
                                    dilation=(1,1),
                                    groups=2)  
        self.bn2=nn.BatchNorm2d(20) 
        self.shortcut1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=20,
                                    kernel_size=(kernel_size,1),
                                    stride=1,
                                    padding=(int((kernel_size-1)/2),0),
                                    dilation=(1,1),
                                    groups=2)
        self.shortcut2=nn.Sequential()
        self.relu = Mish()
    def forward(self,x):
        out=self.relu(self.bn1((self.depth_conv1(x))))
        out=self.depth_conv2(out)
        if self.shortcut==True:
            out=out+self.shortcut1(x)
        else:
            out=out+self.shortcut2(x)
        out=self.relu(self.bn2(out))
        return out
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
      super(SELayer, self).__init__()
     
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.fc = nn.Sequential(
       nn.Linear(channel, channel // reduction, bias=False),
#       nn.ReLU(inplace=True),
       Mish(),
       nn.Linear(channel // reduction, channel, bias=False),
       nn.Sigmoid()
      )
     
    def forward(self, x):
      b, c, _, _ = x.size()
     
      y = self.avg_pool(x).view(b, c)
      y = self.fc(y).view(b, c, 1, 1)
     
      return x * y.expand_as(x)   
#
class resunet(nn.Module):
    def __init__(self):
        super(resunet,self).__init__()
        self.Depth_conv1=Depth_conv(in_channels=2,kernel_size=21,shortcut=True)
        self.Depth_conv2=Depth_conv(in_channels=2,kernel_size=11,shortcut=True)
        self.Depth_conv3=Depth_conv(in_channels=2,kernel_size=5,shortcut=True)
        self.Depth_conv1_1=Depth_conv(in_channels=20,kernel_size=21,shortcut=False)
        self.Depth_conv2_1=Depth_conv(in_channels=20,kernel_size=11,shortcut=False)
        self.Depth_conv3_1=Depth_conv(in_channels=20,kernel_size=5,shortcut=False)
        self.Depth_conv1_2=Depth_conv(in_channels=20,kernel_size=21,shortcut=False)
        self.Depth_conv2_2=Depth_conv(in_channels=20,kernel_size=11,shortcut=False)
        self.Depth_conv3_2=Depth_conv(in_channels=20,kernel_size=5,shortcut=False)
        self.Senet_conv=SELayer(20)
#        self.Point_conv=nn.Conv2d (40, 4, kernel_size=(65,1),
#                                stride=1, padding=(int((65-1)/2),0), bias=False) 
#        self.Point_convBN=nn.BatchNorm2d(4) 
        
        self.layer1=block_down(20,64,kernel_size=(11,1),resnet=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(5,1), stride=(2,1), padding=0)
        self.layer2=block_down(64,64,kernel_size=(5,1),resnet=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        self.layer3=block_down(64,128,kernel_size=(5,1),resnet=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        self.layer4=block_down(128,128,kernel_size=(5,1),resnet=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        
        self.up1= nn.Upsample(  scale_factor=(2,1), mode='bilinear', align_corners=None)  
        self.up1_conv=nn.Conv2d (384, 128, kernel_size=(1,1),
                                stride=1, padding=(0,0), bias=False)        
        self.layer5= block_up(128,128,kernel_size=(5,1),resnet=True)  
        
        self.up2= nn.Upsample(  scale_factor=(2,1), mode='bilinear', align_corners=None)  
        self.layer6= block_up(256,128,kernel_size=(5,1),resnet=False) 
        
        self.up3= nn.Upsample(  scale_factor=(2,1), mode='bilinear', align_corners=None) 
        self.up3_conv=nn.Conv2d (192, 64, kernel_size=(1,1),
                                stride=1, padding=(0,0), bias=False)
        self.layer7= block_up(64,64,kernel_size=(5,1),resnet=True) 
        self.up4= nn.Upsample(  scale_factor=(5,1), mode='bilinear', align_corners=None)  
        self.layer8= block_up(128,64,kernel_size=(11,1),resnet=False) 
        self.conv9=nn.Conv2d (64, 1, kernel_size=(1,1),
                                stride=1, padding=(0,0), bias=False)
        self.PSPModule=PSPModule(128)
        self.LSTM_ATTENTION=LSTM_ATTENTION(40,40,1)
        self.soft=nn.Softmax(dim=1)
        self.sig=nn.Sigmoid()
        self.dense=nn.Linear(125,125)
        self.dropout=Disout(dist_prob=0.2,block_size=6,alpha=30)
        self.globalAvgpool2d=nn.AdaptiveAvgPool2d((1,1))
        self.globalAvgpool2d2=nn.AdaptiveAvgPool2d((1,1))
        self.dense2=nn.Linear(256,256)
        self.dense3=nn.Linear(256,1)
        self.soft2=nn.Sigmoid()
        self.relu = Mish()
    def forward(self,x):
#        depth_conv1=self.Depth_conv4(x)   
#        depth_conv2=self.Depth_conv2(depth_conv1)
#        depth_conv3=self.Depth_conv3(depth_conv2)
        ##新加
#        x=self.Senet_conv(x)
        depth_conv1=self.Depth_conv1(x) 
        depth_conv1_1=self.Depth_conv1_1(depth_conv1)
        depth_conv1_2=self.Depth_conv1_2(depth_conv1_1)
        senet_conv=self.Senet_conv(depth_conv1_2)
#        point_conv=F.relu(self.Point_convBN(self.Point_conv(depth_conv1_2)))
#        depth_conv2=self.Depth_conv2(x)
#        depth_conv2_1=self.Depth_conv2_1(depth_conv2)
#        depth_conv2_2=self.Depth_conv2_2(depth_conv2_1)
#        depth_conv3=self.Depth_conv3(x)
#        depth_conv3_1=self.Depth_conv3_1(depth_conv3)
#        depth_conv3_2=self.Depth_conv3_2(depth_conv3_1)
#        depth_conv4=torch.cat((depth_conv1_2,depth_conv2_2,depth_conv3_2), 1)
        ##
#        print(depth_conv4.size())
        out1=self.layer1(senet_conv)
        
        pool_1=self.pool1(out1)
#        print(pool_1.size())
        out2=self.layer2(pool_1)
        pool_2=self.pool2(out2)
        
        out3=self.layer3(pool_2)
        pool_3=self.pool3(out3)
        
        out4=self.layer4(pool_3)
        pool_4=self.pool4(out4)
#        print("pool_4",pool_4.size())
        
        PSPModule1=self.PSPModule(pool_4)
#        print("PSPModule1",PSPModule1.size())
#        LSTM1=torch.squeeze(PSPModule1, 3)
##        
#        LSTM_output,(h_n,c_n)=self.LSTM_ATTENTION(LSTM1)#h_n,c_n为各个层的最后一个时步的隐藏状态h和c
##        print("LSTM_output",LSTM_output.size())
##        '''
##        attention操作
##        '''
#        attention=self.soft(self.dense(LSTM_output)) 
#        print("attention",attention.size())
#        LSTM_output=LSTM_output*attention
##        print("LSTM_output",LSTM_output.size())
##        print(LSTM_output.size())
#        LSTM_output=torch.unsqueeze(LSTM_output, 3)
#        LSTM_output=self.dropout(LSTM_output)
        
        ##loss2
        globalAvgpool=self.globalAvgpool2d(PSPModule1)
#        print("globalAvgpool2",globalAvgpool.size())
        globalAvgpool=torch.squeeze(globalAvgpool)
        out_softmax=self.dense2(globalAvgpool)
        out_softmax=self.relu(out_softmax)
        out_softmax=self.dense3(out_softmax)
        out_softmax=self.soft2(out_softmax)
        out=torch.unsqueeze(out_softmax, 2)
#        out=torch.unsqueeze(out_softmax, 1)
#        print('out_softmax',out_softmax.size())
        ###
        

#        out=out_softmax.permute(0,2,1)

#        print('out',out.size())
        return out
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.out_channels
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



        
##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=resunet().to(device)

summary(net,(2,1600,1))
##print(net)
#        
#nums=input()
#dp=[0 for _ in range(int(len(nums))+1)]
#aa=loadmat("D:/心梗/2020生理参数挑战赛/其他挑战赛的数据/sig_all_pvc.mat")
#aa=aa['sig_all_pvc']