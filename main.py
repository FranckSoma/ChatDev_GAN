'''
This is the main file for training and testing the GAN model.
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from generator import Generator
from discriminator import Discriminator
from data import get_data_loader
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
torch.manual_seed(42)
# Hyperparameters
batch_size = 64
latent_dim = 100
epochs = 10
# Load MNIST dataset
train_loader = get_data_loader(batch_size, dataset='MNIST')
# Initialize generator and discriminator
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
# Loss function
criterion = nn.BCELoss()
# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# Training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        # Adversarial ground truths
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # Train discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        real_preds = discriminator(real_images)
        fake_preds = discriminator(fake_images.detach())
        d_loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
        d_loss.backward()
        optimizer_D.step()
        # Train generator
        optimizer_G.zero_grad()
        fake_preds = discriminator(fake_images)
        g_loss = criterion(fake_preds, real_labels)
        g_loss.backward()
        optimizer_G.step()
        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
# Save trained models
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), 'models/generator.pth')
torch.save(discriminator.state_dict(), 'models/discriminator.pth')