import torch.nn as nn
import torch.nn.functional as F
import torch
from models.archs.RFMD_arch import RFMD
import torch.nn.init as init


class classSR_2class_RFMD(nn.Module):
    def __init__(self,cfg):
        super(classSR_2class_RFMD, self).__init__()
        self.classifier=Classifier()
        self.net1 = RFMD(cfg)
        self.net2 = RFMD(cfg)

    def forward(self, x, train, scale):
        if train:
            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type, dim=1)
                p1 = p[0][0]
                p2 = p[0][1]
                out1 = self.net1(x[i].unsqueeze(0),scale)
                out2 = self.net2(x[i].unsqueeze(0),scale)
                out = out1 * p1 + out2 * p2

                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)
            return out_res,type_res

        else:
            type = self.classifier(x)
            flag = torch.max(type, 1)[1].data.squeeze()
            p = F.softmax(type, dim=1)
            print(flag)


            if flag == 0:
                out = self.net1(x,scale)
            elif flag == 1:
                out = self.net2(x,scale)

            out_res  = out
            type_res = p

            return out_res, type_res



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 2)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 256, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(256, 512, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(512, 256, 1,1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(256, 128, 1,1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out


