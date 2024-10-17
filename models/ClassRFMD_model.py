import models.networks as networks
import torch.nn as nn
import torch


def Weights_Normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Class_RFMD_Model():
    def __init__(self):
        super(Class_RFMD_Model, self).__init__()

        # define network and load pretrained models
        self.netG = networks.define_G().to(self.device)




