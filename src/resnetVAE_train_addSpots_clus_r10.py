# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tqdm
import torch
import math
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# make torch dataset from tif images
data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_train_addSpots/full'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
max_t = torch.tensor([1.0])
min_t = torch.tensor([0.44313726])
'''train_dist
mean: 0.70216423
std: 0.19584435
max: 1.0
min: 0.44313726
'''

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)+1])
batch_size = 4
bs = 32
zsize = 16

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

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
        self.relu = nn.LeakyReLU(inplace = True)
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
        self.relu = nn.LeakyReLU(inplace=True)
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
        self.relu = nn.LeakyReLU(inplace = True)
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
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
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
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.linear1(x)

        mu = self.linear2(x) # mean
        sigma = torch.exp(self.linear3(x)) # variance
        z = mu + sigma*self.N.sample(mu.shape) ## check this, z has some negative values which will always become 0 cause ReLU
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z

encoder = Encoder(Bottleneck, [3, 4, 6, 3])

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dfc3 = nn.Linear(zsize, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dfc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dfc1 = nn.Linear(64, 64 * 1 * 1)
        self.bn1 = nn.BatchNorm1d(64*1*1)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.dconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv5_1 = nn.Conv2d(32, 32, 3, stride=1, padding='same')
        self.bn_5_2 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(32, 32, 3, stride=1, padding='same')

        self.dconv4 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv4_1 = nn.Conv2d(16, 16, 3, stride=1, padding='same')
        self.bn_4_2 = nn.BatchNorm2d(16)
        self.conv4_2 = nn.Conv2d(16, 16, 3, stride=1, padding='same')

        self.dconv3 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(8, 8, 3, stride=1, padding='same')
        self.bn_3_2 = nn.BatchNorm2d(8)
        self.conv3_2 = nn.Conv2d(8, 8, 3, stride=1, padding='same')

        self.dconv2 = nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(4, 4, 3, stride=1, padding='same')
        self.bn_2_2 = nn.BatchNorm2d(4)
        self.conv2_2 = nn.Conv2d(4, 4, 3, stride=1, padding='same')

        self.dconv1 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)

        self.sigmoidConv = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(True),
            nn.Conv2d(1, 1, 3, stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dfc3(x)
        x = F.relu(self.bn3(x))
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))

        x = x.view(x.shape[0], 64, 1, 1)

        x = self.dconv5(x)
        x = F.relu(self.bn_5_2(self.conv5_1(x)))
        x = F.relu(self.conv5_2(x))

        x = F.relu(self.dconv4(x))
        x = F.relu(self.bn_4_2(self.conv4_1(x)))
        x = F.relu(self.conv4_2(x))

        x = F.relu(self.dconv3(x))
        x = F.relu(self.bn_3_2(self.conv3_1(x)))
        x = F.relu(self.conv3_2(x))

        x = self.dconv2(x)
        x = F.relu(self.bn_2_2(self.conv2_1(x)))
        x = F.relu(self.conv2_2(x))

        x = self.dconv1(x)
        x = self.sigmoidConv(x)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return x, z

## make the model
torch.manual_seed(0)

d = 2

vae = VariationalAutoencoder()
print(vae)
lr = 1e-5

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)

## Training function
def train_epoch(vae, device, dataloader, optimizer, patch_size = 32):
    vae.train()
    train_loss = 0.0
    img_num = 1
    num_samples = 0
    for x, _ in dataloader:
        print(img_num, len(dataloader))
        img_num += 1
        for img in x:
            full_img_patches = []
            for i in range(0, img.shape[1]-patch_size, patch_size):
                for j in range(i, img.shape[2]-patch_size, patch_size):
                    patch_img = img[:, i:i+patch_size, j:j+patch_size]
                    full_img_patches.append(patch_img)
                    if len(full_img_patches) == bs:
                        full_img_patches = torch.stack(full_img_patches)
                        full_img_patches = full_img_patches.to(device)
                        full_img_patches_hat, z_mid = vae(full_img_patches)
                        loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        num_samples += full_img_patches.shape[0]
                        full_img_patches = []
                        # print("hi")

        print('\t partial batch train loss (single batch): %f' % (train_loss / num_samples))

    return train_loss / num_samples

## Testing function
def test_epoch(vae, device, dataloader, patch_size = 32):
    vae.eval()
    val_loss = 0.0
    num_samples_val = 0
    with torch.no_grad():
        for x, _ in dataloader:
            full_img_patches = []
            for img in x:
                for i in range(0, img.shape[1]-patch_size, patch_size):
                    for j in range(i, img.shape[2]-patch_size, patch_size):
                        patch_img = img[:, i:i+patch_size, j:j+patch_size]
                        full_img_patches.append(patch_img)

            full_img_patches = torch.stack(full_img_patches)
            full_img_patches = full_img_patches.to(device)
            full_img_patches_hat, z_mid = vae(full_img_patches)
            loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl

            val_loss += loss.item()
            num_samples_val += full_img_patches.shape[0]
            # print("yo")

    return val_loss / num_samples_val

## plotting function
def plot_ae_outputs(vae_model, n=1, patch_size=32):
    for x, _ in valid_loader:
        img = x[0]

        print(img.shape)

        full_img_patches = []
        for i in range(100, img.shape[1]-patch_size, patch_size):
            for j in range(i, img.shape[2]-patch_size, patch_size):
                patch_img = img[:, i:i+patch_size, j:j+patch_size]
                full_img_patches.append(patch_img)
                if len(full_img_patches) == n:
                    full_img_patches = torch.stack(full_img_patches)
                    break
            break

        full_img_patches = full_img_patches.to(device)
        full_img_patches_hat, z_embed = vae(full_img_patches)

        for s in range(n):
            orig_img = np.squeeze(np.asarray(full_img_patches[s].detach().numpy()), axis=0) * 255
            pred_img = np.squeeze(np.asarray(full_img_patches_hat[s].detach().numpy()), axis=0) * 255
            print(z_embed[s])
            print(full_img_patches_hat[s])
            orig_img = Image.fromarray(orig_img)
            pred_img = Image.fromarray(pred_img)

            fig = plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(orig_img)
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(pred_img)
            plt.show()

        break

# VAE training
num_epochs = 200
best_val_loss = 1e+10
for epoch in range(num_epochs):
    print("Training Epoch {}:".format(epoch+1))
    train_loss = train_epoch(vae, device, train_loader, optim)
    print("Validation")
    val_loss = test_epoch(vae, device, valid_loader)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        print("SAVING")
        best_val_loss = val_loss
        torch.save(vae.state_dict(), 'models/resnetVAE_addSpots_runFULL_10.pt')
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f} \t best_val_loss {:.3f}'.format(epoch+1, num_epochs, train_loss, val_loss, best_val_loss))

# load model and plot outputs
# vae.load_state_dict(torch.load('models/resnetVAE_addSpots_runFULL_1.pt', map_location = torch.device('cpu')))
# vae.eval()
# plot_ae_outputs(vae, n=10)

# try to run testing on the vesicle data.
