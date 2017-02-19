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

parser = argparse.ArgumentParser()

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def roou():
    is_demo = 0

    parser.add_argument("--fwhm", default="1.4")
    parser.add_argument("--sig", default="1.2")
    parser.add_argument("--input_folder", default='./fits_0.01_0.02')
    parser.add_argument("--figure_folder", default="./figures/test")
    args = parser.parse_args()

    train_folder = '%s/train'%(args.figure_folder)
    test_folder = '%s/test'%(args.figure_folder)
    deconv_folder = '%s/deconv'%(args.figure_folder)


    if ~os.path.exists("./figure_folder"):
        os.makedirs("./figure_folder")
    if ~os.path.exists("./train_folder"):
        os.makedirs("./train_folder")
    if ~os.path.exists("./test_folder"):
        os.makedirs("./test_folder")
    if ~os.path.exists("./deconv_folder"):
        os.makedirs("./deconv_folder")

    fits = '%s/*/*-g.fits'%(args.input_folder)
    files = dir(fits)

    for i in files:
        file_name = i

        #readfiles
        if is_demo:
            file_name = '587725489986928743'
            fwhm = 1.4
            sig = 1.2
            mode = 1
            input_folder='fits_0.01_0.02'
            figure_folder='figures'

        filename = filename.replace("-g.fits", '')
        filename_g = '%s/%s/%s-g.fits'%(input_folder,filename,filename)
        filename_r = '%s/%s/%s-r.fits'%(input_folder,filename,filename)
        filename_i = '%s/%s/%s-i.fits'%(input_folder,filename,filename)

        gfits = pyfits.open(filename_g)
        rfits = pyfits.open(filename_r)
        ifits = pyfits.open(filename_i)
        data_g = gfits[0].data
        data_r = rfits[0].data
        data_i = ifits[0].data

        figure_original = np.mat(np.ones(data_g.shape[0],data_g.shape[1],3))
        figure_original[:,:,0] = data_i
        figure_original[:,:,1] = data_r
        figure_original[:,:,2] = data_g

        if is_demo:
            cv2.imshow(figure_original, "img")
            cv2.waitKey(0)

        # gaussian filter
        fwhm_use = fwhm/0.396
        gaussian_sigma = fwhm_use / 2.355
        # with problem
        figure_blurred = cv2.GaussianBlur(figure_original, (5,5), gaussian_sigma)

        if is_demo:
            cv2.imshow(figure_blurred, "i")
            cv2.waitKey(0)

        # add white noise
        figure_original_nz =  figure_original[figure_original<0.1]
        figure_original_nearzero = figure_original_nz[figure_original_nz>-0.1]
        figure_blurred_nz = figure_blurred[figure_blurred<0.1]
        figure_blurred_nearzero = figure_blurred_nz[figure_blurred_nz>-0.1]
        [m,s] = norm.fit(figure_original_nearzero)
        [m2,s2] = norm.fit(figure_blurred_nearzero)

        whitenoise_var = (sig*s)^2-s2^2

        if whitenoise_var < 0:
            whitenoise_var = 0.00000001

        whitenoise = random.normrnd(0, whitenoise_var)
        figure_blurred[:,:,1] = figure_blurred[:,:,1] + whitenoise
        figure_blurred[:,:,2] = figure_blurred[:,:,2] + whitenoise
        figure_blurred[:,:,3] = figure_blurred[:,:,3] + whitenoise

        if is_demo:
            cv2.imshow(figure_blurred,'k')
            cv2.waitKey(0)

        # deconvolution
        if mode:
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
        if mode:
            figure_deconv[figure_deconv<MIN]=MIN
            figure_deconv[figure_deconv>MAX]=MAX

        # normalize figures
        figure_original = (figure_original-MIN)/(MAX-MIN)
        figure_blurred = (figure_blurred-MIN)/(MAX-MIN)
        if mode:
            figure_deconv = (figure_deconv-MIN)/(MAX-MIN)
            if is_demo:
                plt.subplot(1,3,1), plt.subimage(figure_original), plt.subplot(1,3,2), plt.subimage(figure_blurred),plt.subplot(1,3,3), plt.subimage(figure_deconv)
        elif is_demo:
            plt.subplot(1,2,1), plt.subimage(figure_original), plt.subplot(1,2,2), plt.subimage(figure_blurred)

        # asinh scaling
        figure_original = math.asinh(10*figure_original)/3
        figure_blurred = math.asinh(10*figure_blurred)/3

        if mode:
            figure_deconv = math.asinh(10*figure_deconv)/3

        # output result to pix2pix format
        figure_combined = np.zeros(data_g.shape[0], data_g.shape[1]*2,3)
        figure_combined[:,1: data_g.shape[1],:] = figure_original[:,:,:]
        figure_combined[:, data_g.shsape[1]+1:2*data_g.shape[1],:] = figure_blurred[:,:,:]

        if is_demo:
            cv2.imshow(figure_combined)
            cv2.waitKey(0)

        if mode:
            jpg_path = '%s/test/%s.jpg'% (figure_folder,filename)
        else:
            jpg_path = '%s/train/%s.jpg'%(figure_folder,filename)

        if mode == 2:
            figure_combined_no_ori = np.zeros(data_g.shape[0], data_g.shape[1]*2,3)
            figure_combined_no_ori[:, data_g.shape[1]+1:2*data_g.shape[1],:] = figure_blurred[:,:,:]
            cv2.imwrite(figure_combined_no_ori,jpg_path)
        else:
            cv2.imwrite(figure_combined,jpg_path)

        if mode:
            deconv_path = '%s/deconv/%s.jpg'% (figure_folder,filename)
            cv2.imwrite(figure_deconv,deconv_path)


