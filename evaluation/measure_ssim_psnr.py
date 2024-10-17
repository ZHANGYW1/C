from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from imqual_utils import getSSIM, getPSNR


import numpy as np


def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(320, 240)):

    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f==gen_f):
            r_im = Image.open(gtr_path)
            g_im = Image.open(gen_path)
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            r_im = r_im.convert("L"); g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)