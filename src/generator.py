# gan_train.py
import torch.nn as nn

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
num_epochs = 20
batch_size = 100
learning_rate = 0.0002

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)