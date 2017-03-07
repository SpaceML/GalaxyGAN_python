#separate the files into two directories: train, test 
import os
import random 
import glob

path = "/home/chenyiru/fits/*"
train_path = "/mnt/ds3lab/chenyiru/FluxPreservation/fits_train/*"
test_path = "/mnt/ds3lab/chenyiru/FluxPreservation/fits_test/*"
files =  glob.glob(train_path)

random.shuffle(files)
file_number  =  len(files)
print file_number 
for i in range(file_number):
	if i <= 999:
		os.system("ln -s %s %s" % (files[i], "/mnt/ds3lab/chenyiru/FluxPreservation/fits1000"))
	elif i <=1199:
		os.system("ln -s %s %s" % (files[i], "/mnt/ds3lab/chenyiru/FluxPreservation/fits200"))

