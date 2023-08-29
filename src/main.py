import torch
import torch.nn as nn
from torchvision.utils import save_image
from generator import Generator

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained generator
G = Generator().to(device)
G.load_state_dict(torch.load('generator_epoch.pth'))
G.eval()

# Generate and save images
num_samples = 8
z = torch.randn(num_samples, latent_size).to(device)
fake_images = G(z).reshape(-1, 1, 28, 28)  # Reshape to match image dimensions
save_image(fake_images, 'fake_images.png', nrow=8, normalize=True)

print("Generated images saved to 'fake_images.png'")