# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from piqa import SSIM  

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# define VAE
# build the variational auto-encoder
class SSIMLoss(SSIM):
    def forward(self, img1, img2):
        return 1. - super().forward(img1, img2)

class denseBlock(nn.Module):
    def __init__(self):
        super(denseBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
    
    def forward(self, x):
        x = x.to(device)
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = x + x1

        x2 = self.conv1(x1)
        x2 = self.relu(x2)
        x2 = x2 + x1 + x

        x3 = self.conv1(x2)
        x3 = self.relu(x3)
        x3 = x3 + x2 + x1 + x

        x4 = self.conv2(x3)
        x4 = self.relu(x4)

        return x4

class denseBlock_Up(nn.Module):
    def __init__(self):
        super(denseBlock_Up, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = x + x1

        x2 = self.conv1(x1)
        x2 = self.relu(x2)
        x2 = x2 + x1 + x

        x3 = self.conv1(x2)
        x3 = self.relu(x3)
        x3 = x3 + x2 + x1 + x

        x4 = self.conv2(x3)
        x4 = self.relu(x4)
        
        return x4

# build the variational auto-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        
        # first conv
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        
        # second conv
        self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        # this is followed by a leakyrelu
        self.conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # this is followed by a leakyrelu

        self.db1 = denseBlock()
        self.db2 = denseBlock()
        self.db3 = denseBlock()
        self.db4 = denseBlock()
        
        # linear layers
        self.linear1 = nn.Linear(2*2*64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 16) 
        self.linear4 = nn.Linear(16, latent_dims) # mean
        self.linear5 = nn.Linear(16, latent_dims) # variance

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)

        x = self.conv1(x)
        x = F.leaky_relu(self.conv2(x))
        x_inconv = F.leaky_relu(self.conv3(x))
        x1 = self.db1(x_inconv) # dense block 1
        x2 = self.db2(x1) # dense block 2
        x3 = self.db3(x2) # dense block 3
        x4 = self.db4(x3) # dense block 4
        
        x4 = torch.flatten(x4, start_dim=1)

        x_lin1 = F.leaky_relu(self.linear1(x4))
        x_lin2 = F.leaky_relu(self.linear2(x_lin1))
        x_lin3 = F.leaky_relu(self.linear3(x_lin2))

        mu = self.linear4(x_lin3) # mean
        sigma = torch.exp(self.linear5(x_lin3)) # variance
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z, x_inconv, x1, x2, x3, x3, x4, x_lin1, x_lin2, x_lin3

class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.dec_lin1 = nn.Linear(latent_dims, 16) 
        self.dec_lin2 = nn.Linear(16, 64)
        self.dec_lin3 = nn.Linear(64, 128)
        self.dec_lin4 = nn.Linear(128, 2*2*64)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64,2,2)) # 64, 2, 2 = 256

        self.db1_up = denseBlock_Up()
        self.db2_up = denseBlock_Up()
        self.db3_up = denseBlock_Up()
        self.db4_up = denseBlock_Up()

        # first conv
        self.dec_conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        # second conv
        self.dec_conv2 = nn.Conv2d(128, 64, 5, stride=1, padding=2)
        # this is followed by a leakyrelu
        self.dec_conv3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        

    def forward(self, x, x_inconv, x1, x2, x3, x4, x_lin1, x_lin2, x_lin3):
        x = self.dec_lin1(x)
        x = x + x_lin3 

        x = self.dec_lin2(x)
        x = x + x_lin2

        x = self.dec_lin3(x)
        x = x + x_lin1

        x = self.dec_lin4(x)
        x = x + x4 # skip connection 1

        x = self.unflatten(x) # 64, 2, 2 = 256

        x = self.db1_up(x) # dense block Up 1

        x = x + x3 # skip connection 2
        x = self.db2_up(x) # dense block Up 2

        x = x + x2 # skip connection 3
        x = self.db3_up(x) # dense block Up 3

        x = x + x1 # skip connection 4
        x = self.db4_up(x) # dense block Up 4

        x = x + x_inconv # skip connection 5

        x = F.leaky_relu(self.dec_conv1(x)) # first conv
        x = F.leaky_relu(self.dec_conv2(x)) # second conv
        x = torch.sigmoid(self.dec_conv3(x)) # third conv

        x = torch.sigmoid(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z, x_inconv, x1, x2, x3, x3, x4, x_lin1, x_lin2, x_lin3  = self.encoder(x)
        out = self.decoder(z, x_inconv, x1, x2, x3, x4, x_lin1, x_lin2, x_lin3)
        return out, z


# load torch model
d = 4
vae = VariationalAutoencoder(latent_dims = d)
vae.load_state_dict(torch.load('models/DCVAE_r1.pt', map_location = torch.device('cpu')))
vae.eval()

# load test_data
ds = pd.read_csv('../data/bg_data/training_data/bg_remap_total/test_spots_BIG.csv')

data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots_BIG'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)
# test_dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform = transform)

patch_size = 32

def checkPatchSpots(filename, patch_i, patch_j):
    # print(filename)
    # filename = filename.replace('../../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots_BIG/all/', '')
    ds_filename = ds[ds["filename"].isin([filename])]
    x_list = list(ds_filename["spot_x_mid"])
    y_list = list(ds_filename["spot_y_mid"])
    # print(ds_filename)
    for p in range(len(x_list)):
        # print(loc[0], loc[1])
        if ((x_list[p] >= patch_i) and (x_list[p] <= (patch_i + patch_size))) and ((y_list[p] >= patch_j) and (y_list[p] <= (patch_j + patch_size))):
            # print(loc)
            return 1, [x_list[p], y_list[p]] # 1 meaning spot
    return 0, [0, 0] # 0 meaning no spot

embeds_vae = []
img_filename = []
patch_details_i = [] # i to i+32
patch_details_j = [] # j to j+32
spot_details = [] # location of the spot
spots_total_annot = []

for x in range(5):
    print(x)
    sample_x = np.squeeze(train_dataset[x][0].numpy(), axis=0)
    # print(sample_x.shape)
    filename = "../" + train_dataset.imgs[x][0]
    # print(x, len(train_dataset), filename)
    for i in range(0, sample_x.shape[0]-patch_size, patch_size):
        for j in range(i, sample_x.shape[1]-patch_size, patch_size):
            patch_img = sample_x[i:i+patch_size, j:j+patch_size]
            patch_img = np.expand_dims(patch_img, axis=0)
            patch_img = np.expand_dims(patch_img, axis=0)
            # vae = vae.double()
            patch_img = torch.from_numpy(patch_img)
            pred, inter = vae(patch_img)
            # print(inter.shape)
            spots_check, spot_loc = checkPatchSpots(filename, i, j)
            # print(i, j, spots_check, spot_loc)
            # print(i, j, spots_check)
            embeds_vae.append(inter)
            img_filename.append(filename)
            patch_details_i.append(i)
            patch_details_j.append(j)
            spot_details.append(spot_loc)
            spots_total_annot.append(spots_check)
            # print(inter.shape, x, i, j, spot_loc, spots_check)

embeds_vae = np.asarray(torch.stack(embeds_vae).detach().numpy())

csv_dict = {'img_filename': img_filename, 'patch_details_i': patch_details_i, 'patch_details_j': patch_details_j, 'spot_location': spot_details, 'patch_annotation': spots_total_annot}
patches_csv = pd.DataFrame(csv_dict)
print(patches_csv)
patches_csv.to_csv('gen_files/patches_csv_bg_addSpots_4__ONLY1.csv')
np.savez_compressed('gen_files/embeds_DCVAE_addSpots__ONLY1.npz', embeds_vae)
print(patches_csv)
print(embeds_vae.shape)
print(len(spots_total_annot))
'''
sample image test_0 from receptor:
get all the embeds, and the labels
plot all these to see if you can find a separation
'''
