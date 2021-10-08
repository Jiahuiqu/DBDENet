#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:59:59 2020

@author: amax
"""

from torch.utils.data import Dataset
from scipy.io import loadmat
from torch.utils.data import DataLoader
import os
import torch


def SAM(output, HS):
    data1 = torch.sum(output * HS, dim=1)
    # data2 = output.norm(2,dim = 1) * HS.norm(2, dim = 1)
    data2 = torch.sqrt(torch.sum((output ** 2), dim=1) * torch.sum((HS ** 2), dim=1))
    sam_loss = torch.acos((data1 / data2)).view(-1).mean().float()
    # sam_loss = sam_loss.clone().detach().requires_grad_(True)
    return sam_loss


def pic_pour(output, upHS):
    data = (upHS + (upHS * output) / (torch.sum(upHS, dim=1).unsqueeze(1))).float()
    return data


class HS_Dataload(Dataset):
    def __init__(self, root, mode, size):
        super(HS_Dataload, self).__init__()
        self.root = root
        self.mode = mode
        self.size = int(size)
        self.gtHS = []
        self.LRHS = []
        self.PAN = []
        self.UPHS = []
        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root, "train", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.LRHS = os.listdir(os.path.join(self.root, "train", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.PAN = os.listdir(os.path.join(self.root, "train", "PAN"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.I_Spatial = os.listdir(os.path.join(self.root, "train", "I_Spatial"))
            self.I_Spatial.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        if self.mode == "val":
            self.gtHS = os.listdir(os.path.join(self.root, "val", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.LRHS = os.listdir(os.path.join(self.root, "val", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.PAN = os.listdir(os.path.join(self.root, "val", "PAN"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.I_Spatial = os.listdir(os.path.join(self.root, "val", "I_Spatial"))
            self.I_Spatial.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        if self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root, "test", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.LRHS = os.listdir(os.path.join(self.root, "test", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.PAN = os.listdir(os.path.join(self.root, "test", "PAN"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
            self.I_Spatial = os.listdir(os.path.join(self.root, "test", "I_Spatial"))
            self.I_Spatial.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        # print(len(self.gtHS),len(self.LRHS),len(self.PAN),len(self.UPHS))

    def __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):
        gt_hs, lr_hs, pan, I_Spatial = self.gtHS[index], self.LRHS[index], self.PAN[index], self.I_Spatial[index]
        gt_hs_data = loadmat(os.path.join(self.root, self.mode, "gtHS", gt_hs))['da'].reshape(102, self.size, self.size)
        lr_hs_data = loadmat(os.path.join(self.root, self.mode, "LRHS", lr_hs))['dalr'].reshape(102, (self.size // 4),
                                                                                                (self.size // 4))
        I_Spatial = loadmat(os.path.join(self.root, self.mode, "I_Spatial", I_Spatial))['I'].reshape(1,
                                                                                                     (self.size // 4),
                                                                                                     (self.size // 4))
        pan_data = loadmat(os.path.join(self.root, self.mode, "PAN", pan))['dap'].reshape(1, self.size, self.size)
        # up_hs_data = loadmat(os.path.join(self.root,self.mode,"upHS",up_hs))['daup'].reshape(31,self.size,self.size)
        # HRpc1 = get_PC1(up_hs_data,size = up_hs_data.shape[1])
        return lr_hs_data, pan_data, I_Spatial, gt_hs_data


# if __name__ == "__main__":
#     db = HS_Dataload('data', "train", 120)
#     train = DataLoader(db, batch_size=16, )
#     for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(train):
#         print(lr_hs_data.shape, pan_data.shape, I_Spatial.shape, gt_hs_data.shape)