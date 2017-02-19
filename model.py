from config import Config as conf
from utils import conv2d, deconv2d, linear, batch_norm, lrelu
import tensorflow as tf

class CGAN(object):

    def __init__(self):
        self.image = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, conf.img_channel))
        self.cond = tf.placeholder(tf.float32, shape=(1,conf.img_size, conf.img_size, conf.img_channel))

        self.gen_img = self.generator(self.cond)

        pos = self.discriminator(self.image, self.cond, False)
        neg = self.discriminator(self.gen_img, self.cond, True)
        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pos, tf.ones_like(pos)))
        neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(neg, tf.zeros_like(neg)))

        self.d_loss = pos_loss + neg_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(neg, tf.ones_like(neg))) + \
                      conf.L1_lambda * tf.reduce_mean(tf.abs(self.image - self.gen_img))

	t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

    def discriminator(self, img, cond, reuse):
        dim = len(img.get_shape())
        with tf.variable_scope("disc", reuse=reuse):
            image = tf.concat(dim - 1, [img, cond])
            feature = conf.conv_channel_base
            h0 = lrelu(conv2d(image, feature, name="h0"))
            h1 = lrelu(batch_norm(conv2d(h0, feature*2, name="h1"), "h1"))
            h2 = lrelu(batch_norm(conv2d(h1, feature*4, name="h2"), "h2"))
            h3 = lrelu(batch_norm(conv2d(h2, feature*8, name="h3"), "h3"))
            h4 = linear(tf.reshape(h3, [1,-1]), 1, "linear")
        return h4

    def generator(self, cond):
        with tf.variable_scope("gen"):
            feature = conf.conv_channel_base
            e1 = conv2d(cond, feature, name="e1")
            e2 = batch_norm(conv2d(lrelu(e1), feature*2, name="e2"), "e2")
            e3 = batch_norm(conv2d(lrelu(e2), feature*4, name="e3"), "e3")
            e4 = batch_norm(conv2d(lrelu(e3), feature*8, name="e4"), "e4")
            e5 = batch_norm(conv2d(lrelu(e4), feature*8, name="e5"), "e5")
            e6 = batch_norm(conv2d(lrelu(e5), feature*8, name="e6"), "e6")
            e7 = batch_norm(conv2d(lrelu(e6), feature*8, name="e7"), "e7")
            e8 = batch_norm(conv2d(lrelu(e7), feature*8, name="e8"), "e8")

            d1 = deconv2d(tf.nn.relu(e8), [1,2,2,feature*8], name="d1")
            d1 = tf.concat(3, [tf.nn.dropout(batch_norm(d1, "d1"), 0.5), e7])
            d2 = deconv2d(tf.nn.relu(d1), [1,4,4,feature*8], name="d2")
            d2 = tf.concat(3, [tf.nn.dropout(batch_norm(d2, "d2"), 0.5), e6])
            d3 = deconv2d(tf.nn.relu(d2), [1,8,8,feature*8], name="d3")
            d3 = tf.concat(3, [tf.nn.dropout(batch_norm(d3, "d3"), 0.5), e5])
            d4 = deconv2d(tf.nn.relu(d3), [1,16,16,feature*8], name="d4")
            d4 = tf.concat(3, [batch_norm(d4, "d4"), e4])
            d5 = deconv2d(tf.nn.relu(d4), [1,32,32,feature*4], name="d5")
            d5 = tf.concat(3, [batch_norm(d5, "d5"), e3])
            d6 = deconv2d(tf.nn.relu(d5), [1,64,64,feature*2], name="d6")
            d6 = tf.concat(3, [batch_norm(d6, "d6"), e2])
            d7 = deconv2d(tf.nn.relu(d6), [1,128,128,feature], name="d7")
            d7 = tf.concat(3, [batch_norm(d7, "d7"), e1])
            d8 = deconv2d(tf.nn.relu(d7), [1,256,256,conf.img_channel], name="d8")

            return tf.nn.tanh(d8)
