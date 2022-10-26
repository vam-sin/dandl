# libraries
import numpy as np
from PIL import Image
import os

# load receptor data
arr = np.load('../../data/deepBlink_data/receptor.npz', allow_pickle=True)
x_train = arr['x_train']
y_train = arr['y_train']

patch_size = 4

def checkPatchSpots(spots_csv, patch_i, patch_j):
    for loc in spots_csv:
        # print(loc[0], loc[1])
        if ((loc[0] >= patch_i) and (loc[0] <= (patch_i + patch_size))) and ((loc[1] >= patch_j) and (loc[1] <= (patch_j + patch_size))):
            # print(loc)
            return 1, loc
    return 0, [0, 0]

patch_img = []
img_num = []
patch_details_i = [] # i to i+32
patch_details_j = [] # j to j+32
spot_details = [] # location of the spot
spots_total_annot = []

for k in range(len(x_train)):
    sample_x = x_train[k]
    sample_y = y_train[k]
    print(k, len(x_train))
    for i in range(0, sample_x.shape[0]-patch_size, patch_size):
        for j in range(i, sample_x.shape[1]-patch_size, patch_size):
            patch_img = sample_x[i:i+patch_size, j:j+patch_size]
            patch_img = Image.fromarray(patch_img)
            spots_check, spot_loc = checkPatchSpots(sample_y, i, j)
            if spots_check == 0:
                filename = '../../data/deepBlink_data/bg_receptor/all_4/all/img_receptorTrainBG_' + 'imgNum_' + str(k) + '_' + 'i_' + str(i) + '_j_' + str(j) + '.png'
                patch_img.save(filename)


# patch_size = 4
#
# for filename in os.listdir(folder):
#     f = os.path.join(folder, filename)
#     if '.tif' in f:
#         img = Image.open(f)
#         img = np.asarray(img)
#         print(img.shape)
#         for i in range(img.shape[0]-patch_size):
#             for j in range(img.shape[1]-patch_size):
#                 patch_img = img[i:i+patch_size, j:j+patch_size]
#                 print(patch_img.shape)
#                 patch_img = Image.fromarray(patch_img)
#                 patch_img.save('hi.png')
#         break
