import os
import time
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from models.archs.RFMD_arch import RFMD
import yaml
import argparse
## options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="options/rfpm.yaml")
parser.add_argument("--data_dir", type=str, default="the path of testset")
parser.add_argument("--sample_dir", type=str, default="save path")
parser.add_argument("--model_path", type=str, default="the path of checkpoints")
opt = parser.parse_args()

## checks/
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

## model arch
with open(opt.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
scale = cfg["scale"]
model = RFMD(cfg)

## load weights
model.load_state_dict(torch.load(opt.model_path))
if is_cuda: model.cuda()
model.eval()

img_width, img_height, channels = 320, 240, 3
transforms_ = [#transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)


## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = Image.open(path)
    inp_img = inp_img.convert('RGB')
    inp_img = transform(inp_img)
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    s = time.time()
    with torch.no_grad():
        image = model(inp_img,scale)
    times.append(time.time() - s)
    save_image(image, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files))
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))
