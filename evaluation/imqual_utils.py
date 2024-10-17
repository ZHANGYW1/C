from __future__ import division
import numpy as np
import math
from scipy.ndimage import gaussian_filter


def getSSIM(X, Y):
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim==2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5   

    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D 
    mssim = SSIM.mean()

    return mssim

def getPSNR(X, Y):
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0: return 100
    else: return 20*math.log10(255.0/rmse)



