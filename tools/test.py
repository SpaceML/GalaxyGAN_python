from config import Config as conf
from data import *
import scipy.misc
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
import os

def prepocess_test(img, cond):
    #img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    #cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    #img = img/127.5 - 1.
    #cond = cond/127.5 - 1.
    return img,cond

def train():
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    with tf.Session() as sess:
	saver.restore(sess, conf.model_path)
	np_path = "/home/chenyiru/FluxPreservation/demo_out/test/587724648721678356.npy"
	all = np.load(np_path)
	img, cond = all[:,:conf.img_size], all[:,conf.img_size:]

	pimg, pcond = prepocess_test(img, cond)
	gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
	gen_img = gen_img.reshape(gen_img.shape[1:])
	image = np.concatenate((gen_img, cond), axis=1)
	np.save("/home/chenyiru/FluxPreservation/demo_out/587724648721678356.npy", image)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf.use_gpu)
    train()
