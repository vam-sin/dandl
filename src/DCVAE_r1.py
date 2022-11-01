# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tqdm
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

# make torch dataset from tif images
# data_dir = '/nfs_home/nallapar/dandl/data/addSpots/full/'
data_dir = '../data/bg_data/training_data/bg_remap_total/bg_remap_train_addSpots/full'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)+1])
batch_size = 4

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
        return out

## make the model
torch.manual_seed(0)

# model params:
d = 4

# hyperparams:
bs = 16
lr = 1e-5
img_shift = 16

vae = VariationalAutoencoder(latent_dims = d)
print(vae)

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True, patience=30)

criterion = SSIMLoss(n_channels = 1)
# criterion = SSIMLoss(n_channels=1).cuda()

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
        # print(x.shape)
        for img in x:
            # print(img.shape)
            # print(img)
            full_img_patches = []
            for i in range(0, img.shape[1]-patch_size, img_shift):
                for j in range(i, img.shape[2]-patch_size, img_shift):
                    patch_img = img[:, i:i+patch_size, j:j+patch_size]
                    # print(patch_img.shape)
                    full_img_patches.append(patch_img)
                    if len(full_img_patches) == bs:
                        full_img_patches = torch.stack(full_img_patches)
                        # print(full_img_patches.shape)
                        full_img_patches = full_img_patches.to(device)
                        full_img_patches_hat = vae(full_img_patches)
                        # print(full_img_patches.shape, full_img_patches_hat.shape)

                        # SSIM Loss
                        loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl
                        # loss = criterion(full_img_patches, full_img_patches_hat) + vae.encoder.kl

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print('\t partial train loss (single batch): %f' % (loss.item() / full_img_patches.shape[0]))

                        train_loss += loss.item()
                        num_samples += full_img_patches.shape[0]
                        full_img_patches = []

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
                # print(img.shape)
                for i in range(0, img.shape[1]-patch_size, img_shift):
                    for j in range(i, img.shape[2]-patch_size, img_shift):
                        patch_img = img[:, i:i+patch_size, j:j+patch_size]
                        # print(patch_img.shape)
                        full_img_patches.append(patch_img)

            full_img_patches = torch.stack(full_img_patches)
            # print(full_img_patches.shape)
            full_img_patches = full_img_patches.to(device)
            full_img_patches_hat = [] 
            for i in range(0, full_img_patches.shape[0], bs):
                full_img_patches_hat.append(vae(full_img_patches[i:i+bs]))
            
            full_img_patches_hat = torch.stack(full_img_patches_hat)
            # full_img_patches_hat = vae(full_img_patches)
            # print(x.shape, x_hat.shape)

            # MSE Loss
            loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl

            # SSIM Loss
            # loss = criterion(full_img_patches, full_img_patches_hat) + vae.encoder.kl

            val_loss += loss.item()
            num_samples_val += full_img_patches.shape[0]

    return val_loss / num_samples_val

# plotting function
def plot_ae_outputs(vae_model, n=1, patch_size=32):
    vae_model.eval()
    with torch.no_grad():
        for x, _ in valid_loader:
            img = x[0]

            print(img.shape)

            full_img_patches = []
            for i in range(100, img.shape[1]-patch_size, img_shift):
                for j in range(i, img.shape[2]-patch_size, img_shift):
                    patch_img = img[:, i:i+patch_size, j:j+patch_size]
                    full_img_patches.append(patch_img)
                    if len(full_img_patches) == n:
                        full_img_patches = torch.stack(full_img_patches)
                        break
                break

            full_img_patches = full_img_patches.to(device)
            
            full_img_patches_hat = vae_model(full_img_patches)
            
            for s in range(n):
                orig_img = full_img_patches[s]
                pred_img = full_img_patches_hat[s]
                print(full_img_patches_hat[s])

                fig = plt.figure()
                ax1 = fig.add_subplot(2,2,1)
                plt.imshow(orig_img.cpu().squeeze().numpy(), cmap='gist_gray')
                ax2 = fig.add_subplot(2,2,2)
                plt.imshow(pred_img.cpu().squeeze().numpy(), cmap='gist_gray')

                plt.show()

            break

# VAE training
# num_epochs = 500
# best_val_loss = 1e+10
# for epoch in range(num_epochs):
#     print("Training Epoch {}:".format(epoch+1))
#     train_loss = train_epoch(vae, device, train_loader, optim)
#     print("Validation")
#     val_loss = test_epoch(vae, device, valid_loader)
#     scheduler.step(val_loss)
#     if val_loss < best_val_loss:
#         print("SAVING")
#         best_val_loss = val_loss
#         torch.save(vae.state_dict(), 'models/DCVAE_r1.pt')
#     print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f} \t best_val_loss {:.3f}'.format(epoch+1, num_epochs, train_loss, val_loss, best_val_loss))
#     # plot_ae_outputs(vae, n=2)

# plot predictions
vae.load_state_dict(torch.load('models/DCVAE_r1.pt', map_location = torch.device('cpu')))
plot_ae_outputs(vae, n=10)
