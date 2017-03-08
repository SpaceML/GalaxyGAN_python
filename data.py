from config import Config as conf
from utils import imread
import os

def load(path):
    for i in os.listdir(path):
        all = imread(path + "/" + i)
        img, cond = all[:,:conf.img_size], all[:,conf.img_size:]
        yield (img, cond, i)

def load_data():
    data = dict()
    data["train"] = lambda: load(conf.data_path + "/train")
    data["test"] = lambda: load(conf.data_path + "/test")
    return data
