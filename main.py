import torch
import torch.nn as nn
from torch import optim
import os
from scipy.io import loadmat
from model import Spatial_extract
from torch.utils.data import DataLoader
from Dataset import HS_Dataload
from scipy.io import savemat
import Param
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device("cuda:0")
opt = Param.paser.parse_args()
def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(epoch, model, optimizer, criteron, train_loader, show_interview=3):
    model.train()
    loss_meter, count_it = 0, 0
    upsample = nn.Upsample(scale_factor=4, mode="bicubic")
    for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(train_loader):
        LRHS = lr_hs_data.type(torch.float32).to(device)
        PAN = pan_data.type(torch.float32).to(device)
        I_Spatial = I_Spatial.type(torch.float32).to(device)
        gtHS = gt_hs_data.type(torch.float32).to(device)
        out = model(PAN, I_Spatial)
        out = upsample(LRHS) + out
        loss = criteron(out, gtHS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += loss
        count_it += 1
        if step % show_interview == 0:
            print("train-----epoch:", epoch, "step:", step + 1, "loss:", loss.item())
    return float(loss_meter / count_it)

def train():
    db_train = HS_Dataload("data", mode="train", size=opt.img_size)
    train_dataloader = DataLoader(db_train, batch_size=opt.batchsize, shuffle=True, num_workers=32)
    if torch.cuda.is_available():
        print("training start with GPU")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(Spatial_extract()).to(device)
        else:
            model = Spatial_extract.to(device)
    else:
        print("training start with CPU")
        model = Spatial_extract()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = lr)
    optimizer = optim.Adam(model.parameters(), lr= opt.lr, weight_decay = opt.weight_decay)
    criteron = nn.L1Loss()
    best_loss = 10
    for epoch in range(opt.epoch):
        train_loss = train_epoch(epoch,model, optimizer, criteron, train_dataloader)
        if epoch % 1 == 0:
            print("#epoch:%02d best_loss:%0.7f train_loss:%.7f" %(epoch,best_loss, train_loss))
        if train_loss <= best_loss:
            print(train_loss)
            state = dict(epoch = epoch + 1,state_dict = model.state_dict(),best_val = train_loss)
            torch.save(state, "best.mdl")
            best_loss = train_loss
        if (epoch + 1) % 200 == 0:
            opt.lr /= 10
            adjust_learning_rate(opt.lr, optimizer)
    test(model)


def pic_pour(feature, upHS):
    data = upHS + (upHS * feature) / (torch.mean(upHS, dim=1).unsqueeze(1))
    return data

def test(model):
    model.eval()
    checkpoint = torch.load('best.mdl')
    print("min_loss:", checkpoint['best_val'])
    print(checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    db_test = HS_Dataload("data", "test", size=120)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False)
    upsample = nn.Upsample(scale_factor=4, mode="bicubic")
    with torch.no_grad():
        for step, (lr_hs_data, pan_data, I_Spatial, gt_hs_data) in enumerate(test_loader):
            LRHS = lr_hs_data.type(torch.float32).to(device)
            PAN = pan_data.type(torch.float32).to(device)
            I_Spatial = I_Spatial.type(torch.float32).to(device)
            out = model(PAN, I_Spatial)
            out = upsample(LRHS) + out
            filename = "sub//{0}.mat".format(str("out_" + "{0}").format(step + 1))
            savemat(filename, {"data": out.detach().cpu().numpy()})
        print("save success!!!!")


if __name__ == "__main__":
    #train()
