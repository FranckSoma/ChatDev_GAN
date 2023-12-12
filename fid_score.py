'''
This file calculates the FID (Fréchet Inception Distance) score for the GAN model.
'''
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
from generator import Generator
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained generator model
generator = Generator(latent_dim=100).to(device)
generator.load_state_dict(torch.load('models/generator.pth'))
generator.eval()
# Load MNIST dataset
mnist_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
mnist_loader = DataLoader(mnist_dataset, batch_size=100, shuffle=True)
# Generate images using the generator
generated_images = []
with torch.no_grad():
    for _ in range(10):
        z = torch.randn(100, 100).to(device)
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2  # Scale images from [-1, 1] to [0, 1]
        generated_images.append(fake_images)
generated_images = torch.cat(generated_images, dim=0)
# Calculate FID score
def calculate_fid_score(real_images, generated_images):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    real_features = []
    generated_features = []
    with torch.no_grad():
        for real_batch, generated_batch in zip(real_images, generated_images):
            real_features.append(inception_model(real_batch).view(real_batch.size(0), -1))
            generated_features.append(inception_model(generated_batch).view(generated_batch.size(0), -1))
    real_features = torch.cat(real_features, dim=0)
    generated_features = torch.cat(generated_features, dim=0)
    real_mean = torch.mean(real_features, dim=0)
    generated_mean = torch.mean(generated_features, dim=0)
    real_cov = np.cov(real_features.cpu().numpy(), rowvar=False)
    generated_cov = np.cov(generated_features.cpu().numpy(), rowvar=False)
    trace_sqrt = np.trace(sqrtm(real_cov @ generated_cov))
    fid_score = torch.norm(real_mean - generated_mean)**2 + np.trace(real_cov) + np.trace(generated_cov) - 2 * trace_sqrt
    return fid_score.item()
# Calculate FID score
fid_score = calculate_fid_score(mnist_loader, generated_images)
print("FID Score:", fid_score)