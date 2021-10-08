import numpy as np
from scipy import ndimage
import cv2

def cc(img1, img2):
    """CC for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3, "image n_channels should be greater than 1"
    img1_ = np.repeat(np.repeat(np.expand_dims(np.expand_dims(np.mean(img1.reshape(img1.shape[0], -1), axis=1),axis = 1), axis = 2), img1.shape[1], axis = 1), img1.shape[1], axis=2)
    img2_ = np.repeat(np.repeat(np.expand_dims(np.expand_dims(np.mean(img2.reshape(img2.shape[0], -1), axis=1),axis = 1), axis = 2), img2.shape[1], axis = 1), img2.shape[1], axis=2)
    ll = np.sum(np.sum(img1_ * img2_, axis = 1), axis=1)
    img1_d = np.sum(np.sum(img1_ ** 2, axis = 1), axis=1)
    img2_d = np.sum(np.sum(img2_ ** 2, axis = 1), axis=1)
    dd = np.sqrt(img1_d * img2_d)
    return np.mean((ll / dd))

def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))

def RMSE_loss(x1, x2):
    x = x1 - x2
    n, c, h, w = x.shape
    x = torch.pow(x, 2)
    out = torch.sum(x, dim=(1, 2, 3))
    out = torch.pow(torch.div(out, c * h * w), 0.5)
    out = torch.sum(out, 0)
    out = torch.div(out, n)
    return out

def ergas(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == "__main__":
    x = np.random.rand(128,10,10)
    y = np.random.rand(128,10,10)
    print(cc(x, y).shape)