import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_tt_mtl import initialize_weights
import numpy as np


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False,  n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class TOAD_mtl(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes=2):
        super(TOAD_mtl, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [256, 256, 128]}
        #self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        #临床和测序数据的融合
        self.classifier = nn.Linear(size[1] + 9 , n_classes)
        
        # 只是用图像
        #self.classifiers = nn.Linear(size[1], n_classes)
        
        initialize_weights(self)

        print("-------------------------------", initialize_weights)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)
        # self.instance_classifiers = self.instance_classifiers.to(device)


   # def forward(self, h, gender....(所有的融合数据，应该弄个列表）,return_features=False,attention_only=False):
    def forward(self, h,pt,pn,pm,stage,OTOS,LCE3E,PCDHA2,METTL8P1,LINC02124,return_features=False, attention_only=False):
        
        # device = h.device
        #print("--------------------------------",h.shape)
        #h = h.view(h.shape[0] , -1)
        #print("--------------------------------",h.size(0))
        A, h = self.attention_net(h)  # NxK      
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        # gender_tmp = gender.repeat(M.size(0),1)

        #   临床+测序数据的处理             

        tmp =  torch.from_numpy( np.array([pt,pn,pm,stage,OTOS,LCE3E,PCDHA2,METTL8P1,LINC02124], dtype=np.float32) ).to('cuda:0')
       

        # M = torch.mm(A, h)
        M = torch.cat([M, tmp.repeat(M.size(0), 1)], dim=1)
        logits  = self.classifier(M[0].unsqueeze(0))
        #logits = self.classifiers(M)
        
        #return logits
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
            
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})
        return results_dict

