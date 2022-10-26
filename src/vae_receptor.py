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

# make torch dataset from tif images
data_dir = '../data/deepBlink_data/bg_receptor/all_32'

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(data_dir, transform = transform)
# test_dataset = torchvision.datasets.ImageFolder(data_dir + '/test', transform = transform)

m = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)+1])
batch_size = 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# build the variational auto-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.linear1 = nn.Linear(2*2*64, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        # print(x, x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.batch2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.conv4(x))
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
            nn.Linear(128, 2*2*64),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64,2,2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
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
        # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.decoder_conv(x)
        # print(x.shape)
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
        return self.decoder(z)

## make the model
torch.manual_seed(0)

d = 2

vae = VariationalAutoencoder(latent_dims = d)
print(vae)
lr = 1e-3

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)

## Training function
def train_epoch(vae, device, dataloader, optimizer, patch_size = 32):
    vae.train()
    train_loss = 0.0
    num_samples = 0
    for x, _ in dataloader:
        x_hat = vae(x)
        # print(full_img_patches.shape, full_img_patches_hat.shape)
        # print(x.shape, x_hat.shape)
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(x_hat[0], x[0])
        print('\t partial train loss (single batch): %f' % (loss.item() / x.shape[0]))

        train_loss += loss.item()
        num_samples += x.shape[0]

    return train_loss / num_samples

## Testing function
def test_epoch(vae, device, dataloader, patch_size = 32):
    vae.eval()
    val_loss = 0.0
    num_samples_val = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x_hat = vae(x)
            # print(x.shape, x_hat.shape)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()
            num_samples_val += x.shape[0]

    return val_loss / num_samples_val

## plotting function
def plot_ae_outputs(encoder, decoder, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = train_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        img = train_dataset[t_idx[i]][0].unsqueeze(0).to(device)
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
num_epochs = 20
best_val_loss = 1e+10
for epoch in range(num_epochs):
    print("Training:")
    train_loss = train_epoch(vae, device, train_loader, optim)
    print("Validation")
    val_loss = test_epoch(vae, device, valid_loader)
    if val_loss < best_val_loss:
        print("SAVING")
        best_val_loss = val_loss
        torch.save(vae.state_dict(), 'vae_receptor_run10.pt')
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f} \t best_val_loss {:.3f}'.format(epoch+1, num_epochs, train_loss, val_loss, best_val_loss))
    # plot_ae_outputs(vae.encoder, vae.decoder, n=1)

# try to run testing on the receptor data.
