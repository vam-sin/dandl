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

# define VAE
# build the variational auto-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(4*4*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.batch2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
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
            nn.Linear(128, 4*4*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,4,4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding = 1, output_padding=1),
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
        return self.decoder(z), z

# load torch model
d = 2
vae = VariationalAutoencoder(latent_dims = d)
vae.load_state_dict(torch.load('models/vae_bgmimic_remap_run1.pt'))
vae.eval()

# load test_data
ds = pd.read_csv('../data/deepBlink_data/receptor.csv')

arr = np.load('../data/deepBlink_data/receptor.npz', allow_pickle=True)
x_test = arr['x_test']
y_test = arr['y_test']

print(len(x_test), len(y_test))
print(x_test[0].shape, len(y_test[0]))

sample_x = x_test[0]
sample_y = y_test[0]

print(sample_x, sample_y)

patch_size = 32
trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

def checkPatchSpots(spots_csv, patch_i, patch_j):
    for loc in spots_csv:
        # print(loc[0], loc[1])
        if ((loc[0] >= patch_i) and (loc[0] <= (patch_i + patch_size))) and ((loc[1] >= patch_j) and (loc[1] <= (patch_j + patch_size))):
            # print(loc)
            return 1, loc
    return 0, [0, 0]

embeds_vae = []
img_num = []
patch_details_i = [] # i to i+32
patch_details_j = [] # j to j+32
spot_details = [] # location of the spot
spots_total_annot = []

for k in range(len(x_test)):
    sample_x = x_test[k]
    sample_y = y_test[k]
    print(k, len(x_test))
    for i in range(0, sample_x.shape[0]-patch_size, patch_size):
        for j in range(i, sample_x.shape[1]-patch_size, patch_size):
            patch_img = sample_x[i:i+patch_size, j:j+patch_size]
            # print(patch_img.shape)
            patch_img = Image.fromarray(patch_img)
            patch_img = trans(patch_img)
            # patch_img = patch_img / 256
            # patch_img = patch_img.double()
            # patch_img = np.expand_dims(patch_img, axis=0)
            patch_img = np.expand_dims(patch_img, axis=0)
            # print(patch_img)
            # patch_img = Image.fromarray(patch_img)
            # patch_img = trans(patch_img)
            # print(patch_img.shape)
            patch_img = torch.from_numpy(patch_img)
            # patch_img = patch_img.to(device)
            # vae = vae.double()
            pred, inter = vae(patch_img)
            # print(inter.shape)
            spots_check, spot_loc = checkPatchSpots(sample_y, i, j)
            # print(i, j, spots_check)
            embeds_vae.append(inter)
            img_num.append(k)
            patch_details_i.append(i)
            patch_details_j.append(j)
            spot_details.append(spot_loc)
            spots_total_annot.append(spots_check)
            # print(inter.shape, k, i, j, spot_loc, spots_check)

embeds_vae = np.asarray(torch.stack(embeds_vae).detach().numpy())

csv_dict = {'img_number': img_num, 'patch_details_i': patch_details_i, 'patch_details_j': patch_details_j, 'spot_location': spot_details, 'patch_annotation': spots_total_annot}
patches_csv = pd.DataFrame(csv_dict)
# patches_csv.to_csv('patches_csv_receptor.csv')
np.savez_compressed('gen_files/embeds_vae_receptor_fromBGMimic_remap_run1.npz', embeds_vae)
print(patches_csv)
print(embeds_vae.shape)
# print(len(spots_total_annot))
'''
sample image test_0 from receptor:
get all the embeds, and the labels
plot all these to see if you can find a separation
'''
