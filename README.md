# GalaxyGAN_python
This project is the implementation of the Paper "Generative Adversarial Networks recover features in astrophysical images
of galaxies beyond the deconvolution limit" on python.

This repo differs from the code on https://github.com/SpaceML/GalaxyGAN_python/ in the training format -
this repo trains the data as numpy arrays while the other trains as .jpg .

##Setup

##Prerequisites
- Python with numpy
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- TensorFlow 1.0

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/Ireneruru/GalaxyGAN_python.git
```

##Run our code

###Preprocess the .FITs

You can find the roou.py in the directory tools. It is used to preprocess the .fits files.
If the mode equals zero, this is the training data. If the mode equals one, the data is used for testing.

```bash
    python roou.py --input XXX --fwhm XXX --sig XXX --figure XXX --mode 0
```
XXX is your local address.

Then modify the constants in the Config.py.

###Train the model

```bash
    python train.py
```

### Tools

There are still some other codes in tools: plot.py, test.py, visualize.py etc.

To highlight, the train.py has tested the data in the test dataset after saving a model. If you want to test other data, you can run the test.py.
Remember there are some constants in test.py you need to modify.


##Acknowledge

This project is for the space.ml project.
