import yaml
import argparse
import torch
from PIL import Image
from models.dataloader.image_utils import torchPSNR
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.archs.RFMD_arch import RFMD
from models.loss import VGG19_PercepLoss
from tqdm import tqdm
from torchvision.utils import save_image
from models.dataloader.data_utils import GetTrainingPairs, GetValImage
import warnings
from adamp import AdamP
warnings.filterwarnings("ignore", category=UserWarning)

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="options/rfpm.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=301, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")

args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["img_width"]
img_height = cfg["img_height"]
val_interval = cfg["val_interval"]
scale = cfg["scale"]

L1_G  = torch.nn.L1Loss() # similarity loss (l1) # content loss (vgg)
L_vgg = VGG19_PercepLoss() # content loss (vgg)
generator = RFMD(cfg)


# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

#load pretrained models
if args.epoch != 0:
    generator.load_state_dict(torch.load("/generator_%d.pth" % (dataset_name, args.epoch)))
    print ("Loaded model from epoch %d" %(epoch))


# Optimizers
optimizer_G = AdamP(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2), weight_decay=1e-2)


## Data pipeline
transforms_ = [
    #transforms.Resize((img_height, img_width), Image.BICUBIC),
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
    GetValImage(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = 4,
    shuffle = True,
    num_workers = 1,
)

if __name__ == '__main__':
    ## Training pipeline
    best_psnr = 0
    generator.train()
    for epoch in range(epoch, num_epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            # Model inputs
            imgs_good_gt = Variable(batch["A"].type(Tensor))
            imgs_distorted = Variable(batch["B"].type(Tensor))
            ## Train Generator
            optimizer_G.zero_grad()
            imgs_fake = generator(imgs_distorted,scale)
            loss_1 = L1_G(imgs_fake, imgs_good_gt) + 0.01 * L_vgg(imgs_fake, imgs_good_gt) # similarity loss
            loss_G = loss_1
            loss_G.backward()
            optimizer_G.step()

            ##print log
            if not i % 50:
                sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [GLoss: %.3f]"
                                 % (epoch, num_epochs, i, len(dataloader),loss_G.item()))

        if epoch % val_interval == 0:
           PSNRs = []
           generator.eval()
           for ii, batch in enumerate(tqdm(val_dataloader)):
               imgs_val = Variable(batch["val"].type(Tensor))
               imgs_val_tg = Variable(batch["val_tg"].type(Tensor))
               with torch.no_grad():
                    imgs_gen = generator(imgs_val,scale)
                    for res, tar in zip(imgs_gen, imgs_val_tg):
                        psnr = torchPSNR(res, tar)
                        PSNRs.append(psnr)
           PSNRs = torch.stack(PSNRs).mean().item()

           if PSNRs > best_psnr:
              print(best_psnr)
              best_psnr = PSNRs
              best_epoch_psnr = epoch
              torch.save (generator.state_dict(), "./checkpoints/generator.pth")
           print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr))






