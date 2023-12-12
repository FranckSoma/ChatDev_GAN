'''
This file defines the user interface for the GAN model.
'''
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from generator import Generator
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained generator model
generator = Generator(latent_dim=100).to(device)
generator.load_state_dict(torch.load('models/generator.pth'))
generator.eval()
# Generate and save images
z = torch.randn(10, 100).to(device)
fake_images = generator(z)
fake_images = (fake_images + 1) / 2  # Scale images from [-1, 1] to [0, 1]
# Save generated images
save_image(fake_images, 'generated_images.png', nrow=10)
print ("Images have been created")