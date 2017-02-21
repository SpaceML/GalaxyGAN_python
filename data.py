from config import Config as conf
from utils import imread
import os

def load(path):
    imgs = []
    for i in os.listdir(path):
        all = imread(path + "/" + i)
        img, cond = all[:,:conf.img_size], all[:,conf.img_size:]
        imgs.append((img, cond))
    return imgs

def load_data():
    data = dict()
    data["train"] = load(conf.data_path + "/train")
    # data["val"] = load(conf.data_path + "/val")
    data["test"] = load(conf.data_path + "/test")
    return data




