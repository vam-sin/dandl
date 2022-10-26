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

# make torch dataset from tif images
data_dir = '/nfs_home/nallapar/dandl/data/addSpots/full/'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)+1])
batch_size = 4

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# build the variational auto-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding='same')
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding='same')
        self.batch2 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding='same')
        self.batch3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding='same')
        self.batch4 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding='same')
        self.batch5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding='same')
        self.batch6 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(4*4*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)

        x = F.leaky_relu(self.batch1(self.conv1(x)))
        x = F.leaky_relu(self.batch2(self.conv2(x)))
        x = F.leaky_relu(self.maxpool1(x))

        x = F.leaky_relu(self.batch3(self.conv3(x)))
        x = F.leaky_relu(self.batch4(self.conv4(x)))
        x = F.leaky_relu(self.maxpool2(x))

        x = F.leaky_relu(self.batch5(self.conv5(x)))
        x = F.leaky_relu(self.batch6(self.conv6(x)))
        x = F.leaky_relu(self.maxpool3(x))

        x = torch.flatten(x, start_dim=1)

        x = F.leaky_relu(self.linear1(x))
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
            nn.LeakyReLU(True),
            nn.Linear(128, 4*4*32),
            nn.LeakyReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,4,4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding = 1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 16, 3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 8, 3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

        self.sigmoidConv = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)

        x = self.sigmoidConv(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

## make the model
torch.manual_seed(0)

# model params:
d = 16

# hyperparams:
bs = 32
lr = 1e-3

vae = VariationalAutoencoder(latent_dims = d)
print(vae)

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
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
        # print(x.shape)
        for img in x:
            # print(img.shape)
            # print(img)
            full_img_patches = []
            for i in range(0, img.shape[1]-patch_size, patch_size):
                for j in range(i, img.shape[2]-patch_size, patch_size):
                    patch_img = img[:, i:i+patch_size, j:j+patch_size]
                    # print(patch_img.shape)
                    full_img_patches.append(patch_img)
                    if len(full_img_patches) == bs:
                        full_img_patches = torch.stack(full_img_patches)
                        # print(full_img_patches.shape)
                        full_img_patches = full_img_patches.to(device)
                        full_img_patches_hat = vae(full_img_patches)
                        # print(full_img_patches.shape, full_img_patches_hat.shape)
                        loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl

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
                for i in range(0, img.shape[1]-patch_size, patch_size):
                    for j in range(i, img.shape[2]-patch_size, patch_size):
                        patch_img = img[:, i:i+patch_size, j:j+patch_size]
                        # print(patch_img.shape)
                        full_img_patches.append(patch_img)

            full_img_patches = torch.stack(full_img_patches)
            # print(full_img_patches.shape)
            full_img_patches = full_img_patches.to(device)
            full_img_patches_hat = vae(full_img_patches)
            # print(x.shape, x_hat.shape)
            loss = ((full_img_patches - full_img_patches_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()
            num_samples_val += full_img_patches.shape[0]

    return val_loss / num_samples_val

## plotting function
def plot_ae_outputs(encoder, decoder, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 +n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images')
        plt.show()

# VAE training
num_epochs = 100
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
        torch.save(vae.state_dict(), 'models/start_vae_addSpots_run4.pt')
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f} \t best_val_loss {:.3f}'.format(epoch+1, num_epochs, train_loss, val_loss, best_val_loss))
    # plot_ae_outputs(vae.encoder, vae.decoder, n=2)

# try to run testing on the vesicle data.
