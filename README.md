# GalaxyGAN_python
This project is the implementation of the Paper "Generative Adversarial Networks recover features in astrophysical images of galaxies beyond the deconvolution limit" on python. It is the python version of https://github.com/SpaceML/GalaxyGAN. This python version doesn't include deconvolution part of the paper.

## Amazon EC2 Setup

### EC2 Public AMI
We provide an EC2 AMI with the following pre-installed packages:

* CUDA
* cuDNN
* Tensorflow r0.12
* python

as well as the FITS file we used in the paper(saved in ~/fits_train and ~/fits_test)

AMI Id: ami-96a97f80
    . (Can be launched using p2.xlarge instance in GPU compute catagory)

 [Launch](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Images:sort=visibility) an instance.
### Connect to Amazon EC2 Machine

Please follow the instruction of Amazon EC2.

## Prerequisites

Linux or OSX

NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

## Dependencies

We need the following python packages:
`tensorflow`, `cv2`, `numpy`, `scipy`, `matplotlb`, `pyfits`, and `ipython`

## Get Our Code    
Clone this repo:

```bash
git clone https://github.com/SpaceML/GalaxyGAN_python.git 
cd  GalaxyGAN_python/
```

## Get Our FITS Files
The data to download is about 5GB, after unzipping it will become about 16GB. Download this file from Google Drive: https://drive.google.com/open?id=1GCs02NBnr7X3skA04hyuXh6cUMZQLzVe


## Run Our Code


### Preprocess the .FITs
If the mode equals zero, this is the training data. If the mode equals one, the data is used for testing.

```bash
python roou.py --input fitsdata/fits_train --fwhm 1.4 --sig 1.2 --mode 0
python roou.py --input fitsdata/fits_test --fwhm 1.4 --sig 1.2 --mode 1
```
XXX is your local address. On our AMI, you can skip this step due to all these have default values.


### Train the model

If you need, you can modify the constants in the Config.py.

```bash
python train.py gpu=1
```
You can appoint which gpu to run the code by changing "gpu=1".

This will start the training process. If you want to load the model which already exists, you can modify the model_path in the config.py.

### Test 

Before you try to test your model, you should modify the model path in the config.py. 

```bash 
python test.py gpu=1
```
The results can be seen in the folder "test".
