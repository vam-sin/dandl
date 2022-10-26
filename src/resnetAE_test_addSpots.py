# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import math
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# hyperparams
batch_size = 4
bs = 32
zsize = 32

# resnet based variational autoencoder
def conv3x3(in_planes, out_planes, stride=1):
    '''
    3x3 conv with padding
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self, block, layers, latent_dims = zsize):
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 4, 1024)
        self.linear1 = nn.Linear(1024, zsize)
        self.linear2 = nn.Linear(zsize, latent_dims)
        self.linear3 = nn.Linear(zsize, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes *block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.linear1(x)
        # print(x.shape)
        return x

encoder = Encoder(Bottleneck, [3, 4, 6, 3])
# encoder.fc = nn.Linear(2048, zsize)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dfc3 = nn.Linear(zsize, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dfc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dfc1 = nn.Linear(1024, 1024 * 1 * 1)
        self.bn1 = nn.BatchNorm1d(1024*1*1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
        self.dconv4 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dconv1 = nn.ConvTranspose2d(128, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # print(x)
        x = self.dfc3(x)
        x = F.relu(self.bn3(x))
        # x = F.relu(x)
        # print(x)
        # print("SPLIT")
        # print(x.shape)

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # x = F.relu(x)
        # print(x)
        # print("SPLIT")
        # print(x.shape)

        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # x = F.relu(x)
        # print(x)
        # print("SPLIT")
        # print(x.shape)

        x = x.view(x.shape[0], 1024, 1, 1)
        # print(x.shape)
        # print(x)
        # print("SPLIT")

        # x = self.upsample1(x)
        # print(x)
        # print("SPLIT")
        # print(x.shape)

        x = self.dconv5(x)
        # print(x.shape)
        # x = self.upsample1(x)
        x = F.relu(self.dconv4(x))
        # print(x.shape)
        x = F.relu(self.dconv3(x))
        # print(x.shape)
        # x = self.upsample1(x)
        # print(x.shape)
        x = self.dconv2(x)
        # print(x.shape)
        x = F.relu(x)
        # x = self.upsample1(x)
        # print(x)
        # print(x.shape)
        x = self.dconv1(x)
        # print(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, x):
        # print(x[0], x[0].shape, x.max(), x.min())
        z = self.encoder(x)
        # print(x.shape)
        out = self.decoder(z)
        # print(x[0], x[0].shape)

        return out, z

## make the model
torch.manual_seed(0)

vae = VariationalAutoencoder()
print(vae)

vae.load_state_dict(torch.load('models/resnetAE_addSpots_runFULL_1.pt', map_location = torch.device('cpu')))
vae.eval()

# load test_data
ds = pd.read_csv('../data/bg_data/training_data/bg_remap_total/test_spots.csv')

data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_test_addSpots'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)

patch_size = 32

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

for x in range(10):
    print(x)
    sample_x = np.squeeze(test_dataset[x][0].numpy(), axis=0)
    # print(sample_x.shape)
    filename = "../" + test_dataset.imgs[x][0]
    # print(x, len(test_dataset), filename)
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
    print(len(embeds_vae))

embeds_vae = np.asarray(torch.stack(embeds_vae).detach().numpy())

csv_dict = {'img_filename': img_filename, 'patch_details_i': patch_details_i, 'patch_details_j': patch_details_j, 'spot_location': spot_details, 'patch_annotation': spots_total_annot}
patches_csv = pd.DataFrame(csv_dict)
print(patches_csv)
patches_csv.to_csv('gen_files/patches_csv_bg_addSpots_32__O1.csv')
np.savez_compressed('gen_files/embeds_resnetAE_BG_run1__O1.npz', embeds_vae)
print(patches_csv)
print(embeds_vae.shape)
print(len(spots_total_annot))
'''
sample image test_0 from receptor:
get all the embeds, and the labels
plot all these to see if you can find a separation
'''
