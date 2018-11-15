# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Params(object):
    n_epochs = 50
    batch_size = 128
    lr = 0.0002
    hidden_size = 256
    img_size = 28
    img_channels = 1
    n_labels = 10
    embedding_size = 100
    sample_interval = 500


class Generator(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.embedding = nn.Embedding(p.n_labels, p.embedding_size)
        self.model = nn.Sequential(
            nn.Linear(p.hidden_size + p.embedding_size, p.hidden_size),
            nn.BatchNorm1d(p.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(p.hidden_size, p.img_size ** 2),
            nn.LeakyReLU(),
        )

    def forward(self, noise, label):
        img = self.model(torch.cat([noise, self.embedding(label)], -1))
        img = img.view(img.size(0),  # batch size
                       self.p.img_channels,
                       self.p.img_size,
                       self.p.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.embedding = nn.Embedding(p.n_labels, p.embedding_size)
        self.model = nn.Sequential(
            nn.Linear(p.img_channels * p.img_size ** 2 + p.embedding_size, p.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(p.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(torch.cat([img_flat, self.embedding(label)], -1))
        return validity


def sample_image(generator, n_row, noise_dim, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    noise = torch.randn(n_row ** 2, noise_dim).to(DEVICE)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels).to(DEVICE)
    gen_imgs = generator(noise, labels)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)


def main():
    p = Params()
    # Loss function
    adversarial_loss = torch.nn.BCELoss().to(DEVICE)
    # Initialize generator and discriminator
    generator = Generator(p).to(DEVICE)
    discriminator = Discriminator(p).to(DEVICE)

    os.makedirs('./data/mnist', exist_ok=True)
    os.makedirs('./images', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0, 0, 0), (1, 1, 1))
                       ])),
        batch_size=p.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=p.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=p.lr)

    for epoch in range(p.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(DEVICE)
            fake = torch.zeros(imgs.size(0), 1).to(DEVICE)
            # Sample noise as generator input
            noise = torch.randn(imgs.shape[0], p.hidden_size).to(DEVICE)
            real_imgs = torch.Tensor(imgs).to(DEVICE)
            labels = labels.to(DEVICE)

            #  Train Discriminator
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            gen_imgs = generator(noise, labels)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
            d_loss = real_loss + fake_loss
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            #  Train Generator
            optimizer_G.zero_grad()
            # Sample noise as generator input
            noise = torch.randn(imgs.shape[0], p.hidden_size).to(DEVICE)
            gen_imgs = generator(noise, labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            optimizer_G.step()

            print("[Epoch: {:3d}/{:3d}] [Batch {:3d}/{:3d}] [D loss: {:.4f}] [G loss: {:.4f}]"
                  .format(epoch, p.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % p.sample_interval == 0:
                sample_image(generator, n_row=10, noise_dim=p.hidden_size, batches_done=batches_done)


if __name__ == '__main__':
    main()
