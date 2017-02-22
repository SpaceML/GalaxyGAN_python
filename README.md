# GalaxyGAN_python
This project is the implementation of the Paper "Generative Adversarial Networks recover features in astrophysical images of galaxies beyond the deconvolution limit" on python.

##Setup

###EC2 Public AMI
We provide an EC2 AMI with the following pre-installed packages:

* CUDA
* cuDNN
* Tensorflow r0.12
* python

as well as the FITS file we used in the paper(saved in ~/fits_train and ~/fits_test)

AMI Id: ami-96a97f80
    . (Can be launched using p2.xlarge instance in GPU compute catagory)

 [Launch](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Images:sort=visibility) an instance.
###Connect to Amazon EC2 Machine
    Please follow the instruction of Amazon EC2.

##Get our code    
- Clone this repo:
```bash
    git clone https://github.com/SpaceML/GalaxyGAN_python.git 
```

##Run our code
After you launch a instance,first

```bash
    cd  GalaxyGAN_python/
```

###Preprocess the .FITs
If the mode equals zero, this is the training data. If the mode equals one, the data is used for testing.

```bash
    python roou.py -input XXX -fwhm 1.4 -sigma 1.2 -figure XXX -gpu 1 -model models -mode 0
```
XXX is your local address. On our AMI, you can skip this step due to all these have default values.


###Train the model

If you need, you can modify the constants in the Config.py.
```bash
    python train.py
```
This will begin to train the model. If you want to load the model which already exists, you can modify the model_path in the config.py.

###Test 

Before you try to test your model, you should modify the model path in the config.py. 
```bash 
    python test.py
```
The results can be seen in the folder "test".

##Acknowledge

This project is the python version of https://github.com/SpaceML/GalaxyGAN. Thanks for his work!
