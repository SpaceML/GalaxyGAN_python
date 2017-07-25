#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import cv2
import math
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pyfits
import glob
from IPython import embed
# mode : 0 training : 1 testing

parser = argparse.ArgumentParser()

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def adjust(origin):
    img = origin.copy()
    img[img>4] = 4
    img[img < -0.1] = -0.1
    MIN = np.min(img)
    MAX = np.max(img)
    img = np.arcsinh(10*(img - MIN)/(MAX-MIN))/3
    return img

def roou():
    is_demo = 0

    parser.add_argument("--fwhm", default="1.4")
    parser.add_argument("--sig", default="1.2")
    parser.add_argument("--input", default="/home/ubuntu/GalaxyGAN_python/fits_train")   #./fits_0.01_0.02
    parser.add_argument("--figure", default="figures")       #./figures/test
    parser.add_argument("--gpu", default = "1")
    parser.add_argument("--model", default = "models")
    parser.add_argument("--mode", default="0")
    args = parser.parse_args()

    fwhm = float(args.fwhm)
    sig  = float(args.sig)
    input =  args.input
    figure = args.figure
    mode = int(args.mode)

    train_folder = '%s/train'%(args.figure)
    test_folder = '%s/test'%(args.figure)
    deconv_folder = '%s/deconv'%(args.figure)

    if not os.path.exists('./' + args.figure):
        os.makedirs("./" + args.figure)
    if not os.path.exists("./" + train_folder):
        os.makedirs("./" + train_folder)
    if not os.path.exists("./" + test_folder):
        os.makedirs("./" + test_folder)
    #if not os.path.exists("./" + deconv_folder):
        #os.makedirs("./"+ deconv_folder)

    fits = '%s/*/*-g.fits'%(input)
    files = glob.iglob(fits)

    for i in files:
        #print i
        file_name = os.path.basename(i)

        #readfiles
        if False:
            file_name = '587725489986928743'
            fwhm = 1.4
            sig = 1.2
            mode = 1
            input_folder='fits_0.01_0.02'
            figure_folder='figures'

        filename = file_name.replace("-g.fits", '')
        filename_g = '%s/%s/%s-g.fits'%(input,filename,filename)
        filename_r = '%s/%s/%s-r.fits'%(input,filename,filename)
        filename_i = '%s/%s/%s-i.fits'%(input,filename,filename)

        gfits = pyfits.open(filename_g)
        rfits = pyfits.open(filename_r)
        ifits = pyfits.open(filename_i)
        data_g = gfits[0].data
        data_r = rfits[0].data
        data_i = ifits[0].data

        figure_original = np.ones((data_g.shape[0],data_g.shape[1],3))
        figure_original[:,:,0] = data_g
        figure_original[:,:,1] = data_r
        figure_original[:,:,2] = data_i

        #print figure_original

        if is_demo:
            cv2.imshow("img", adjust(figure_original))
            cv2.waitKey(0)

        # gaussian filter
        fwhm_use = fwhm/0.396
        gaussian_sigma = fwhm_use / 2.355

        # with problem
        figure_blurred = cv2.GaussianBlur(figure_original, (5,5), gaussian_sigma)

        #print "IIIIII"

        if is_demo:
            cv2.imshow("i", figure_blurred)
            cv2.waitKey(0)

        # add white noise
        figure_original_nz =  figure_original[figure_original<0.1]
        figure_original_nearzero = figure_original_nz[figure_original_nz>-0.1]
        figure_blurred_nz = figure_blurred[figure_blurred<0.1]
        figure_blurred_nearzero = figure_blurred_nz[figure_blurred_nz>-0.1]
        [m,s] = norm.fit(figure_original_nearzero)
        [m2,s2] = norm.fit(figure_blurred_nearzero)

        whitenoise_var = (sig*s)**2-s2**2

        if whitenoise_var < 0:
            whitenoise_var = 0.00000001

        whitenoise =  np.random.normal(0, np.sqrt(whitenoise_var) , (data_g.shape[0], data_g.shape[1]))

        figure_blurred[:,:,0] = figure_blurred[:,:,0] + whitenoise
        figure_blurred[:,:,1] = figure_blurred[:,:,1] + whitenoise
        figure_blurred[:,:,2] = figure_blurred[:,:,2] + whitenoise

        if is_demo:
            cv2.imshow('k',figure_blurred)
            cv2.waitKey(0)

        # deconvolution
        if mode>2:
            hsize = 2*np.ceil(2*gaussian_sigma)+1
            PSF = fspecial_gauss(hsize, gaussian_sigma)
            #figure_deconv = deconvblind(figure_blurred, PSF)
            if is_demo:
                cv2.imshow(figure_deconv)
                cv2.waitKey(0)

        # thresold
        MAX = 4
        MIN = -0.1

        figure_original[figure_original<MIN]=MIN
        figure_original[figure_original>MAX]=MAX

        figure_blurred[figure_blurred<MIN]=MIN
        figure_blurred[figure_blurred>MAX]=MAX
        if mode>2:
            figure_deconv[figure_deconv<MIN]=MIN
            figure_deconv[figure_deconv>MAX]=MAX

        # normalize figures
        figure_original = (figure_original-MIN)/(MAX-MIN)
        figure_blurred = (figure_blurred-MIN)/(MAX-MIN)

        '''if mode:
            figure_deconv = (figure_deconv-MIN)/(MAX-MIN)
            if is_demo:
                plt.subplot(1,3,1), plt.subimage(figure_original), plt.subplot(1,3,2), plt.subimage(figure_blurred),plt.subplot(1,3,3), plt.subimage(figure_deconv)
        elif is_demo:
            plt.subplot(1,2,1), plt.subimage(figure_original), plt.subplot(1,2,2), plt.subimage(figure_blurred)
        '''

        # asinh scaling
        figure_original = np.arcsinh(10*figure_original)/3
        figure_blurred = np.arcsinh(10*figure_blurred)/3

        if mode>2:
            figure_deconv = np.arcsinh(10*figure_deconv)/3

        # output result to pix2pix format
        figure_combined = np.zeros((data_g.shape[0], data_g.shape[1]*2,3))
        figure_combined[:,: data_g.shape[1],:] = figure_original[:,:,:]
        figure_combined[:, data_g.shape[1]:2*data_g.shape[1],:] = figure_blurred[:,:,:]

        if is_demo:
            cv2.imshow(figure_combined)
            cv2.waitKey(0)

        if mode:
            jpg_path = '%s/test/%s.jpg'% (figure,filename)
        else:
            jpg_path = '%s/train/%s.jpg'% (figure,filename)

        if mode == 2:
            figure_combined_no_ori = np.zeros(data_g.shape[0], data_g.shape[1]*2,3)
            figure_combined_no_ori[:, data_g.shape[1]:2*data_g.shape[1],:] = figure_blurred[:,:,:]
            cv2.imwrite(figure_combined_no_ori,jpg_path)
        else:
            image = (figure_combined*256).astype(np.int)
            cv2.imwrite(jpg_path, image)

        if mode>2:
            deconv_path = '%s/deconv/%s.jpg'% (figure_folder,filename)
            cv2.imwrite(figure_deconv,deconv_path)

roou()
