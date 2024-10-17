import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from models.networks import define_G
import yaml

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="path of testset")
parser.add_argument("--sample_dir", type=str, default="save path")
parser.add_argument("--cfg_file", type=str, default="options/class_rfpm.yaml", help="the path of option file")
opt = parser.parse_args()

with open(opt.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
mode = cfg["train"]
scale = cfg["scale"]

## checks/
#assert exists(opt.model_path), "model not found"
#os.makedirs(opt.sample_dir, exist_ok=True)

model = define_G(cfg)
model.load_state_dict(torch.load("checkpoints"))
is_cuda = torch.cuda.is_available()
if is_cuda:
   model.cuda()
   Tensor = torch.cuda.FloatTensor
else:
   Tensor = torch.FloatTensor

model.eval()

transforms_ = [transforms.Resize((240, 320), Image.BICUBIC),
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
    # generate enhanced image
    s = time.time()
    model.eval()
    with torch.no_grad():
        time_start = time.time()
        image,type = model(inp_img,mode,scale)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
    times.append(time.time() - s)
    save_image(image, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files))
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))



