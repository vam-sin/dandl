# libraries
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import random

# # load data
mypath = '../../data/bg_data/training_data/bg_remap_total/bg_remap_test/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

high_int = [x for x in range(245, 256)]
# print(high_int)
start_ = [x for x in range(1200-10)]
# print(start_)

filename_list = []
sp_list_x_mid = []
sp_list_y_mid = []

spot_size = 15

for i in onlyfiles:
    if '.tif' in i:
        filename = mypath + i
        print(filename)
        img = Image.open(filename)
        np_img = np.asarray(img).copy()
        # np_img.setflags(write=1)
        # print(np_img.shape)
        filename_save = '../../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots_BIG/all/' + i
        for sp in range(100): # 100 ribosome spots
            #plt.imshow(img)
            #plt.show()
            # print(np_img[100][100])
            start_x_sp = random.choice(start_)
            start_y_sp = random.choice(start_)
            for x in range(start_x_sp, start_x_sp+spot_size):
                for y in range(start_y_sp, start_y_sp+spot_size):
                    if x < np_img.shape[0] and y < np_img.shape[1]:
                        np_img[x][y] = random.choice(high_int)
                        filename_list.append(filename_save)
                        sp_list_x_mid.append(start_x_sp + spot_size/2)
                        sp_list_y_mid.append(start_y_sp + spot_size/2)
            # print(np_img[100][100])
        img2 = Image.fromarray(np_img)
        # plt.imshow(img2)
        # plt.show()
            # img = img.convert('L')
            # plt.imshow(img)
            # plt.show()
        filename_save = '../../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots_BIG/all/' + i
        img2.save(filename_save)
            # break
        # break
dict_csv = {'filename': filename_list, 'spot_x_mid': sp_list_x_mid, 'spot_y_mid': sp_list_y_mid}
spots_csv = pd.DataFrame(dict_csv)
spots_csv.to_csv('../../data/bg_data/training_data/bg_remap_total/test_spots_BIG.csv')
