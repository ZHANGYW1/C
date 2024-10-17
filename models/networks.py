import torch
import models.archs.class_RFMD_arch



# # Generator
def define_G(cfg):

    netG = models.archs.class_RFMD_arch.classSR_2class_RFMD(cfg)

    return netG