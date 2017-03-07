from config import Config as conf
from data import *
import scipy.misc
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
import os

def prepocess_train(img, cond):
    #img = scipy.misc.imresize(img, [conf.adjust_size, conf.adjust_size])
    #cond = scipy.misc.imresize(cond, [conf.adjust_size, conf.adjust_size])
    #h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    #w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.adjust_size)))
    #img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    #cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]

    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)

    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    return img,cond

def prepocess_test(img, cond):
    #img = scipy.misc.imresize(img, [conf.train_size, conf.train_size])
    #cond = scipy.misc.imresize(cond, [conf.train_size, conf.train_size])
    img = img.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    cond = cond.reshape(1, conf.train_size, conf.train_size, conf.img_channel)
    #img = img/127.5 - 1.
    #cond = cond/127.5 - 1.
    return img,cond

def train():
    

    data = load_data()
    model = CGAN()

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)

    start_epoch = 0
    try:
        log = open(conf.save_path + "/log")
        start_epoch = int(log.readline())
        log.close()
    except:
        pass

    with tf.Session() as sess:
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
        for epoch in xrange(start_epoch, conf.max_epoch):
            train_data = data["train"]()
            for img, cond, _ in train_data:
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M, flux = sess.run([g_opt, model.g_loss, model.delta], feed_dict={model.image:img, model.cond:cond})
                counter += 1
                print "Iterate [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f, flux: %.8f"\
                      % (counter, time.time() - start_time, m, M, flux)
            if (epoch + 1) % conf.save_per_epoch == 0:
                save_path = saver.save(sess, conf.save_path + "/model.ckpt")
                print "Model saved in file: %s" % save_path

                log = open(conf.save_path + "/log", "w")
                log.write(str(epoch + 1))
                log.close()

                test_data = data["test"]()
                test_count = 0
                for img, cond, name in test_data:
                    test_count += 1
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])
                    image = np.concatenate((gen_img, cond), axis=1)
                    np.save(conf.output_path + "/" + name,image)

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf.use_gpu)
    train()
