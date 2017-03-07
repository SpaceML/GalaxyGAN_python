import numpy as np
import cv2
import os
import glob
import sys

#used the lambda = 10
visualize_input_path = sys.argv[1]
visualize_save_dir = sys.argv[2]
if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

fits = "%s/*.npy"%(visualize_input_path)
files =  glob.iglob(fits)

for i in files:
    print i
    fits = np.load(i)
    file_name = os.path.basename(i)
    filename = file_name.replace(".npy", '.jpg')

    # thresold
    #MAX = 4
    #MIN = -0.1

    #fits[fits<MIN]=MIN
    #fits[fits>MAX]=MAX

    #normalize figures
    #fits = (fits-MIN)/(MAX-MIN)

    #asinh scaling
    # fits = np.arcsinh(10*fits)/3

    visualize_save_path = '%s/%s'% (visualize_save_dir,filename)

    #integerization
    image = (fits*256).astype(np.int)
    cv2.imwrite(visualize_save_path, image)
