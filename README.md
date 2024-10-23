# Dependencies and Installation #
1. Python >= 3.8 (Recommend to use Anaconda)
2. PyTorch >= 1.12.1
3. Option: NVIDIA GPU + CUDA

# Train and Inference #
## train RFMD ## 
1. Set the path of dataset.
2. Run train_RFMD.py
## test RFMD ## 
1. Set the path of testset.
2. You can download the pretrained model of SESR(upscaling factor = 2) :
https://drive.google.com/file/d/1hZiuDeoOfX7Lxs_KA5XxXdb9an6nLAvZ/view?usp=drive_link
4. Run test_RFMD.py
## train SRFMD ##
1. In file train_RFMD.py, from models.archs.SRFMD_arch import SRFMD, and replace the RFMD with SRFMD.
2. Set the path of dataset.
3. Run train_RFMD.py
##  test SRFPM ##
1. In file test_RFMD.py, from models.archs.SRFMD_arch import SRFMD, and replace the RFMD with SRFMD.
2. Set the path of testset.
3. Run test_RFMD.py
## train Class_RFMD ## 
1. Set the mode in the options/rfpm.yaml.
mode = True: Training process
mode = False: Testing Process
2. Set the path of the dataset.
3. Run train_Class_RFMD.py.
## test Class_RFMD ##
1. Set the mode in the options/rfpm.yaml.
2. You can download the pretrained model of SESR (upscaling factor = 2) with classfication module:
https://drive.google.com/file/d/1BRYwRtdk1TtUIOfjcJE5PxxCiXYF0kBb/view?usp=drive_link
4. Set the path of testset
5. Run test_Class_RFMD.py
# Acknowledge #
The repo is partly built based on [FUnIEGAN](https://github.com/xahidbuffon/FUnIE-GAN), [URIE](https://github.com/taeyoungson/urie), [Class SR](https://github.com/XPixelGroup/ClassSR), [ConvMLP](https://github.com/SHI-Labs/Convolutional-MLPs). We are grateful for their generous contribution to open source.
