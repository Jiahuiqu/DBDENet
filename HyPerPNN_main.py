#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:41:24 2020

@author: amax
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:12:17 2020

@author: amax
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:39:47 2020

@author: amax
"""

import torch
import torch.nn as nn
from torch import optim
import os
from scipy.io import loadmat
from HyPerPNN import HyperPNN
from torch.utils.data import DataLoader
from Dataset import HS_Dataload
from scipy.io import savemat
import numpy as np
import random
device = torch.device("cuda:0")


def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def RSME_loss(x1, x2):
    x = x1 - x2
    n, c, h, w = x.shape
    x = torch.pow(x, 2)
    out = torch.sum(x, dim=(1, 2, 3))
    out = torch.pow(torch.div(out, c * h * w), 0.5)
    out = torch.sum(out, 0)
    out = torch.div(out, n)
    return out


def SAM(output, HS):
    data1 = torch.sum(output * HS, dim=1)
    # data2 = output.norm(2,dim = 1) * HS.norm(2, dim = 1)
    data2 = torch.sqrt(torch.sum((output ** 2), dim=1) * torch.sum((HS ** 2), dim=1))
    sam_loss = torch.acos((data1 / data2)).view(-1).mean().type(torch.float32)
    sam_loss = sam_loss.clone().detach().requires_grad_(True)
    return sam_loss


def CC(output, HS):
    out_u = torch.mean(output, dim=[2, 3]).unsqueeze(2).unsqueeze(3).repeat(1, 1, output.shape[2], output.shape[2])
    HS_u = torch.mean(output, dim=[2, 3]).unsqueeze(2).unsqueeze(3).repeat(1, 1, output.shape[2], output.shape[2])
    ll = (out_u * HS_u).sum(dim=[2, 3])
    out_d = (out_u ** 2).sum(dim=[2, 3])
    HS_d = (HS_u ** 2).sum(dim=[2, 3])
    dd = (out_d * HS_d).sqrt()
    cc_loss = (ll / dd).view(-1).mean()
    return cc_loss


def train_epoch(epoch, model, optimizer, criteron, train_loader, show_interview=3):
    model.train()
    loss_meter, count_it = 0, 0
    for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(train_loader):
        LRHS = lr_hs_data.type(torch.float32).to(device)
        PAN = pan_data.type(torch.float32).to(device)
        I_Spatial = I_Spatial.type(torch.float32).to(device)
        gtHS = gt_hs_data.type(torch.float32).to(device)
        out = model(LRHS, PAN)
        loss = criteron(out, gtHS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += loss
        count_it += 1
        # if step % show_interview == 0:
        #     print("train-----epoch:", epoch, "step:" ,step + 1, "loss:",loss.item())
    return float(loss_meter / count_it)



def val_epoch(epoch, model, criteron, val_loader, show_interview=3):
    model.eval()
    loss_meter, count_it = 0, 0
    with torch.no_grad():
        down_sample = nn.Upsample(scale_factor=0.5, mode="bilinear")
        up_sample_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        up_sample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(val_loader):
            LRHS = lr_hs_data.type(torch.float32).to(device)
            PAN = pan_data.type(torch.float32).to(device)
            I_Spatial = I_Spatial.type(torch.float32).to(device)
            gtHS = gt_hs_data.type(torch.float32).to(device)
            out1, out2 = model(PAN, I_Spatial)
            interpolation_LRHS_2 = up_sample_2(LRHS)
            interpolation_LRHS_4 = up_sample_4(LRHS)
            down_REF_2 = down_sample(gtHS)
            out1 = pic_pour(out1.repeat(1, 102, 1, 1), interpolation_LRHS_2)
            out2 = pic_pour(out2.repeat(1, 102, 1, 1), interpolation_LRHS_4)
            loss = 0.1 * criteron(out1, down_REF_2) + criteron(out2, gtHS)
            loss_meter += loss
            count_it += 1
            if step % show_interview == 0:
                print("#val-----epoch:", epoch, "step:", step + 1, "loss:", loss.item())
    return float(loss_meter / count_it)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(max_epoch, batchsz, lr):
    set_seed(1)
    db_train = HS_Dataload("data", mode="train", size=120)
    train_dataloader = DataLoader(db_train, batch_size=batchsz, shuffle=True, num_workers=8, pin_memory=True)
    model = nn.DataParallel(HyperPNN()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteron = nn.L1Loss()
    best_loss = 10
    for epoch in range(max_epoch):
        train_loss = train_epoch(epoch, model, optimizer, criteron, train_dataloader)
        # val_loss = val_epoch(epoch,model,criteron,val_dataloader)
        if train_loss <= best_loss:
            # print(train_loss)
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=train_loss)
            torch.save(state, "best.mdl")
            best_loss = train_loss
        if epoch % 1 == 0:
            print("#epoch:%02d best_loss:%.7f" % (epoch, best_loss))
    test(model)


def pic_pour(feature, upHS):
    data = upHS + (upHS * feature) / (torch.mean(upHS, dim=1).unsqueeze(1))
    return data


def test(model):
    model.eval()
    checkpoint = torch.load('best.mdl')
    print("min_loss:", checkpoint['best_val'])
    model.load_state_dict(checkpoint['state_dict'])
    db_test = HS_Dataload("data", "test", size=120)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False)
    with torch.no_grad():
        up_sample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(test_loader):
            LRHS = lr_hs_data.type(torch.float32).to(device)
            PAN = pan_data.type(torch.float32).to(device)
            I_Spatial = I_Spatial.type(torch.float32).to(device)
            output = model(LRHS, PAN)
            filename = "sub//{0}.mat".format(str("out_" + "{0}").format(step + 1))
            savemat(filename, {"data": output.detach().cpu().numpy()})
        print("save success!!!!")


if __name__ == "__main__":
    train(500, 8, 0.0001)




