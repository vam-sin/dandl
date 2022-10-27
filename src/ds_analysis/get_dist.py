# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tqdm
import torch
import math
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# make torch dataset from tif images
data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_train_addSpots'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)
# test_dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform = transform)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)+1])
batch_size = 1
bs = 32
zsize = 48

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

full_vals = []

for x, _ in train_loader:
    print(x.shape)
    x = x.numpy()
    # print(x.shape)
    x = np.squeeze(x, axis=0)
    x = np.squeeze(x, axis=0)
    # print(x.shape)
    for i in range(len(x)):
        for j in range(len(x)):
            full_vals.append(x[i][j])

print(np.mean(full_vals), np.std(full_vals), np.max(full_vals), np.min(full_vals))
