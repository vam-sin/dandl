# libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# # load data
mypath = '../../data/bg_data/training_data/foreground/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in onlyfiles:
    if '.tif' in i:
        filename = mypath + i
        print(filename)
        img = Image.open(filename)
        # plt.imshow(img)
        # plt.show()
        img = img.convert('L')
        # plt.imshow(img)
        # plt.show()
        filename_save = '../../data/bg_data/training_data/fg_remap/' + i
        img.save(filename_save)

# check foreground
# img = Image.open('../../data/bg_data/training_data/foreground/20220505_2xTO9xsm_3h_1.tif')
# img = img.convert('L')
# plt.imshow(img)
# plt.show()
#
# img2 = Image.open('../../data/bg_data/training_data/background/train/all/20220503_2xTO9xsm_0h_2.tif')
# img2 = img2.convert('L')
# plt.imshow(img2)
# plt.show()
