# libraries
import numpy as np
import pandas as pd
from PIL import Image

'''
check the csv files: 🧑‍💻
they have info for three subsets where they described their frame name, signal to noise ratio, and the density.
'''
ds = pd.read_csv('../../data/deepBlink_data/microtubule.csv')
# print(ds)

'''
check the npz files
keys in the npz file:
x_train
x_valid
x_test
y_train
y_valid
y_test

each image is (512, 512) and the y is a list of the (x,y) coordinates
of the different ribosomes.
'''
arr = np.load('../../data/deepBlink_data/receptor.npz', allow_pickle=True)
x_test = arr['x_train']
y_test = arr['y_train']

print(len(x_test), len(y_test))
print(x_test[0].shape, len(y_test[0]))

# img = Image.fromarray(x_test[0], 'L')
# # img.save('images/x_test_0_vesicle.png')
# img.show()
