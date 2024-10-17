import torch
import torch.nn as nn
import torch.nn.functional as F
from .src.modules import ConvMLPStage_S
from basicsr.ops.fused_act import fused_leaky_relu
import math
from torch.nn import Linear
import numpy as np
from torchvision import models


class SRFMD(nn.Module):
    def __init__(self,cfg):
        super(SRFMD, self).__init__()
        scale = cfg["scale"]
        img_width = cfg["img_width"]
        img_height = cfg["img_height"]
        N = int(512 * (img_height // 16) * (img_width // 16))
        self.N1_IMG = N2(out_chans=scale**2*3)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.vgg19 = VGG19_PercepLoss()
        self.linear = Linear(N, 512)

    def forward(self,x):
        xx = self.Upsample(x)
        vgg_feature = self.vgg19(x)
        b = vgg_feature.shape[0]
        vgg_feature = vgg_feature.view(b,-1)
        vgg_feature = self.linear(vgg_feature)
        p2= self.N1_IMG(x, vgg_feature)
        IMG_en = torch.add(p2, xx)

        return IMG_en

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features.cuda()
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, true):
        layer = 'conv5_2'
        true_f = self.get_features(true)
        return true_f[layer]

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):

        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        torch.cuda.manual_seed(1)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out

class FC_8(nn.Module):
    def __init__(self,
                 input_channels=512,
                 out_channls=512,
                 normalize_latents=True,
                 use_wscale=True,
                 lrmul=0.01,
                 gain=2 ** (0.5)
                 ):
        super(FC_8, self).__init__()
        self.input_channls = input_channels
        self.func = nn.Sequential(
            FC(self.input_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(out_channls, out_channls, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):

        torch.cuda.manual_seed(2)
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)

        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class N2(nn.Module):
    def __init__(self, bilinear=True, out_channels1=3):
        super(N2, self).__init__()
        self.bilinear = bilinear

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv1 = nn.Conv2d(3,4,kernel_size=1,padding=0)
        self.Conv= nn.Conv2d(4, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=4, bias=False)
        self.down2 = SKDown1(3, 1, False, 16, 32, 64,32,512)
        self.down3 = SKDown1(3, 1, False, 16, 64, 64,64,512)
        self.up1 = SKUp(3, 1, False, 16, 128, 32,128,512, bilinear=False)
        self.up2 = SKUp(3, 1, False, 16, 64, 32, 64,512, bilinear=False)
        self.up4 = nn.Conv2d(kernel_size=3, padding=1, in_channels=32, out_channels=out_channels1)

    def forward(self, x,vgg_feature):
        x1= self.Conv1(x)
        x1 = self.Conv(x1)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        p1 = self.up1(x3, x2,vgg_feature)
        p2 = self.up2(p1, x1,vgg_feature)
        p3 = self.up4(p2)
        return p3

class select(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(select, self).__init__()
        self.condition_scale1 = EqualLinear(in_channels, out_channels, bias=True, bias_init_val=1, activation=None)

    def forward(self, x):
        scale1,scale2 = self.condition_scale1(x)

        return scale1,scale2

class EqualLinear(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul
        torch.cuda.manual_seed(3)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        torch.cuda.manual_seed(4)
        self.weight1 = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
            out1  = F.linear(x, self.weight1 * self.scale, bias=bias)

        return out, out1

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, bias={self.bias is not None})')

#### Thanks to Taeyoung Son
#### Code from https://github.com/taeyoungson/urie/tree/master/models

class Selector(nn.Module):
    def __init__(self, channel, reduction=16, crp_classify=False):
        super(Selector, self).__init__()
        self.spatial_attention = 4
        self.in_channel = channel * (self.spatial_attention ** 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.spatial_attention, self.spatial_attention))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):

        b, c, H, W = x.size()

        y = self.avg_pool(x).view(b, -1)
        y = self.fc(y)

        att1 = self.att_conv1(y).view(b, c, self.spatial_attention, self.spatial_attention)
        att2 = self.att_conv2(y).view(b, c, self.spatial_attention, self.spatial_attention)

        attention = torch.stack((att1, att2))
        attention = nn.Softmax(dim=0)(attention)

        att1 = F.interpolate(attention[0], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        att2 = F.interpolate(attention[1], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")


        return att1, att2


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):

        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class SelectiveConv(nn.Module):
    def __init__(self,kernel_size, padding, bias, reduction, in_channels, out_channels,embedding_dim, dim_feedforward, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels=in_channels, stride = 2, kernel_size=2, padding=0, bias=False)
        self.selector = Selector(out_channels, reduction=reduction)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.ConvMLPStage =ConvMLPStage_S(embedding_dim, dim_feedforward=dim_feedforward, stochastic_depth_rate=0.1)
        self.ConvMLPStage1 = ConvMLPStage_S(embedding_dim, dim_feedforward=dim_feedforward, stochastic_depth_rate=0.1)
        self.select = select(in_channels=1, out_channels=out_channels)
        self.norm = Norm2Scale()
        self.PixelNorm =  PixelNorm()
        self.FC_8 = FC_8()
        self.linear = FC(512, int(out_channels), gain=1.0, use_wscale = False)
        self.ApplyNoise = ApplyNoise(int(out_channels/2))


    def forward(self, x,vgg_feature):
        if self.first:
            s_input = x
        else:
            # short distance feature or local feature
            x_L1 = self.conv4(x)
            x_L2 = self.conv4(x)
            x_L3 = self.conv4(x)
            x_L4 = self.conv4(x)
            x_L = torch.cat((x_L1,x_L2,x_L3,x_L4),dim=0)
            x1 = x.unfold(2, 2, 2).unfold(3, 2, 2)
            x1 = x1.reshape(x.shape[0], x.shape[1], int(x1.shape[2]), int(x1.shape[3]), -1)
            # long distance sparse feature or gobal feature
            x1_1 = x1[:, :, :, :, 0]
            x1_2 = x1[:, :, :, :, 1]
            x1_3 = x1[:, :, :, :, 2]
            x1_4 = x1[:, :, :, :, 3]
            x1 = torch.cat((x1_1, x1_2, x1_3, x1_4), dim=0)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.ConvMLPStage(x1)
            x1 = x1.permute(0, 3, 1, 2)
            xx =  x1 + x_L
            B= xx.shape[0]//4
            xx_1,xx_2,xx_3,xx_4 = torch.split(xx,[B,B,B,B],dim=0)
            xx= torch.cat((xx_1,xx_2,xx_3,xx_4),dim=1)
            ps = nn.PixelShuffle(2)
            x1 = ps(xx)
            s_input = self.IN(x)
            s_input = self.relu(s_input)
        out0 = self.conv3(x1)
        out0, x12 = torch.split(out0, [int(out0.shape[1]/2),int(out0.shape[1]/2)], dim=1)
        randn_A = self.FC_8(vgg_feature)
        style = self.linear(randn_A)
        x12 = self.ApplyNoise (x12)
        shape = [x12.size(0), 2, x12.size(1), 1, 1]
        style = style.view(shape)
        x12 = x12 * (style[:, 0] + 1.) + style[:, 1]
        out0 = torch.cat((out0,x12),dim=1)
        #scale module
        scale1,scale2 =self.select(torch.ones(1).cuda())
        scale1, scale2 =abs(scale1),abs(scale2)
        scale1, scale2 = self.norm(scale1, scale2)
        scale1 = scale1.view(-1, out0.size(1), 1, 1)
        out2 = self.conv2(s_input)
        scale2 = scale2.view(-1, out0.size(1), 1, 1)

        out0 = out0 * scale1
        out2 = out2 * scale2
        out = out0 + out2
        #attention module
        att0, att2 = self.selector(out)
        out = torch.mul(out0, att0) + torch.mul(out2, att2)

        return out

class Norm2Scale(nn.Module):
    def forward(self, scale1, scale2):
        scales_norm = scale1**2 + scale2**2 + 1e-8
        return scale1 * torch.rsqrt(scales_norm), scale2 * torch.rsqrt(scales_norm)


class SKDown1(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels,embedding_dim, dim_feedforward,first=False):
        super(SKDown1, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels,embedding_dim, dim_feedforward, first=first)
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SKUp(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels,embedding_dim, dim_feedforward, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels, embedding_dim, dim_feedforward)

    def forward(self, x1, x2,vgg_feature):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='trunc'), torch.div(diffX - diffX, 2, rounding_mode='trunc'),
                        torch.div(diffY, 2, rounding_mode='trunc'), torch.div(diffX - diffX, 2, rounding_mode='trunc')])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x,vgg_feature)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass


from torch import nn

