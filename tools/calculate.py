import numpy as np
import scipy.misc
import sys
import os

original = sys.args[1]
testres_path = sys.args[2]
flux_path = sys.args[3]

def imread(path):
    scipy.misc.imread(path)

def load(path):
    print path
    for i in os.listdir(path):
	all = np.load(path +'/'+i)
	img, cond = all[:,:conf.img_size], all[:,conf.img_size:]
	yield(img,cond,i)

def Psnr(im1,im2):
    diff = np.abs(im1 - im2).astype(np.float32)
    mse = np.square(diff).mean()
    psnr = 20*np.log10(255)- 10*np.log10(mse)
    return psnr

def difmean(im1, im2):
    dif = np.square(np.mean(im1) - (np.mean(im2)))
    return dif

def find(name):
    data = dict()
    data["res"] = lambda :load(testres_path)
    for img, cond, n in data["res"]():
        if name == n:
            return img
    return None

def calculate():
    data = dict()
    data["original"] = lambda :load(original)
    # data["val"] = load(conf.data_path + "/val")
    f = open(flux_path, "w")
    for img, cond, name in data["original"]():
        res = find(name)
        if res == None:
            print name, "is not found"
        flux = difmean(img, res)
        psnr = Psnr(img, res)
        f.write(name + "\tpsnr:\t" + str(psnr) +"\tflux:\t"+ str(flux) + "\n")
    f.close()
    return data

calculate()

