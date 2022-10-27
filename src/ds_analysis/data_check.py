# libraries
import numpy as np
import skimage.color
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('../data/bg_data/training_data/background/test/all/20220512_18xsm_0h_3.tif').convert('L')

# folder = '../data/bd_data/training_data/background/test'
# image = skimage.io.imread(fname = '../data/bg_data/training_data/background/test/all/20220512_18xsm_0h_3.tif', as_gray=True)
# fig, ax = plt.subplots()
plt.imshow(img, cmap = "gray")
img = np.asarray(img)/256
# print(np.asarray(img)/256)
histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))

print(histogram)

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)
