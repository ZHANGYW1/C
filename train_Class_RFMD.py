import yaml
import argparse
from PIL import Image
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.ClassRFMD_model import Weights_Normal
from models.networks import define_G
from models.dataloader.data_utils import GetTrainingPairs, GetValImage
from models.loss import class_loss_2class,average_loss_2class
import sys
from torchvision.utils import save_image

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
parser.add_argument("--cfg_file", type=str, default="options/class_rfpm.yaml", help="the path of option file")

args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2

with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

ckpt_interval = cfg["ckpt_interval"]
img_height = cfg["img_height"]
img_width = cfg["img_width"]
mode = cfg["train"]
scale = cfg["scale"]
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]
branchA = cfg["branchA"]
branchB = cfg["branchB"]

generator = define_G(cfg)

if torch.cuda.is_available():
    generator = generator.cuda()
    L1_G  = torch.nn.L1Loss()
    cri_pix = nn.L1Loss().cuda()
    class_loss = class_loss_2class().cuda()
    average_loss = average_loss_2class().cuda()
    # similarity loss (l1) # content loss (vgg)
    Tensor = torch.cuda.FloatTensor

transforms_ = [
   # transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 1,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

pre_state_dict_branch1 = torch.load(branchA)
pre_state_dict_branch2 = torch.load(branchB)
network1 = generator.net1
network2 = generator.net2

load_net_clean = OrderedDict()  # remove unnecessary 'module.'
for k, v in pre_state_dict_branch1.items():
    if k.startswith('module.'):
        load_net_clean[k[7:]] = v
    else:
        load_net_clean[k] = v
network1.load_state_dict(load_net_clean, strict=True)

load_net_clean = OrderedDict()
for k, v in pre_state_dict_branch2.items():
    if k.startswith('module.'):
        load_net_clean[k[7:]] = v
    else:
        load_net_clean[k] = v
network2.load_state_dict(load_net_clean, strict=True)

# fix the parameters of the enhancement module
optim_params = []
for k, v in generator.named_parameters():  # can optimize for a part of the model
    if v.requires_grad and "class" not in k:
        v.requires_grad = False
for k, v in generator.named_parameters():  # can optimize for a part of the model
    if v.requires_grad:
        optim_params.append(v)

optimizer_G = torch.optim.Adam(optim_params, lr=lr_rate, betas=(lr_b1, lr_b2))


if args.epoch == 0:
    generator.classifier.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("path"))

if __name__ == '__main__':
    ## Training pipeline
    generator.train()
    network1.eval()
    network2.eval()
    for epoch in range(epoch, num_epochs):
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_good_gt = Variable(batch["A"].type(Tensor))
            imgs_distorted = Variable(batch["B"].type(Tensor))

            optimizer_G.zero_grad()
            img_fake, type = generator(imgs_distorted,mode,scale)
            loss_G = 4 * cri_pix(img_fake, imgs_good_gt) + 0.5 * class_loss(type) + 3 * average_loss(type)
            loss_G.backward()
            optimizer_G.step()


            if not i % 50:
                sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [GLoss: %.3f]"
                                 % (epoch, num_epochs, i, len(dataloader),loss_G.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % val_interval == 0:
                generator.eval()
                imgs = next(iter(val_dataloader))
                imgs_val = Variable(imgs["val"].type(Tensor))
                imgs_gen,type = generator(imgs_val,mode,scale)
                save_image(imgs_gen, "samples/%s.jpg" % (batches_done), nrow=1,
                           normalize=True)


        ## Save model checkpoints
        if (epoch % ckpt_interval == 0):
           torch.save(generator.state_dict(), "checkpoints/generator_%d.pth" % (epoch))





