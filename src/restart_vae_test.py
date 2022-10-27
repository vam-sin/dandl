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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# define VAE
# build the variational auto-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x) # mean
        sigma = torch.exp(self.linear3(x)) # variance
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,3,3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# load torch model
d = 16
vae = VariationalAutoencoder(latent_dims = d)
vae.load_state_dict(torch.load('models/restart_vae_addSpots.pt'))
vae.eval()

# load test_data
ds = pd.read_csv('../data/bg_data/training_data/bg_remap_total/test_spots.csv')

data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)
# test_dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform = transform)

patch_size = 4

def checkPatchSpots(filename, patch_i, patch_j):
    # print(filename)
    filename = filename.replace('../../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots/all/', '')
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

for x in range(1):
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
            # im_orig = Image.fromarray(patch_img[0])
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

# embeds_vae = np.asarray(torch.stack(embeds_vae).detach().numpy())
#
# csv_dict = {'img_filename': img_filename, 'patch_details_i': patch_details_i, 'patch_details_j': patch_details_j, 'spot_location': spot_details, 'patch_annotation': spots_total_annot}
# patches_csv = pd.DataFrame(csv_dict)
# print(patches_csv)
# patches_csv.to_csv('gen_files/patches_csv_bg_addSpots_4__ONLY1.csv')
# np.savez_compressed('gen_files/embeds_vae_BGAddSpots_run1__ONLY1.npz', embeds_vae)
# print(patches_csv)
# print(embeds_vae.shape)
# print(len(spots_total_annot))
'''
sample image test_0 from receptor:
get all the embeds, and the labels
plot all these to see if you can find a separation
'''
