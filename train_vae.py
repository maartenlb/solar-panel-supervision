import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import SolarObject, collate_fn

# import torch.nn.functional as f
import matplotlib.pyplot as plt
import os.path
import json
import torch.nn.functional as F

# Model Hyperparameters

epochs = 10000

seed = 42
batch_size = 1
lr = 0.00001
loss_factor = 100  # weight of recon loss between 0 and 1

resize = 200

latent_dim = 4096
hidden_dim = 247 * 7 * 7


# Seed everything
torch.manual_seed(seed)
np.random.seed(seed)

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
DEVICE = torch.device("cpu")

transforms = transforms.Compose(
    [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5051, 0.5083, 0.4524], std=[0.1720, 0.1332, 0.1220]
        ),
    ]
)

kwargs = {"num_workers": 0, "pin_memory": True}

# Load your custom dataset
dataset = SolarObject(
    data_dir="data/processed_imgs/",
    json_path="data/image_polygons.json",
    transform=transforms,
    random_seed=seed,
)


# Create data loaders
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=dataset.train_sampler(),
)
val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=dataset.val_sampler(),
)
test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=dataset.test_sampler(),
)


# Help functions
class Pleaseprint(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


# class Inter(nn.Module):
#     def __init__(self, channel, height, width):
#         super(Inter, self).__init__()
#         self.channel = channel
#         self.height = height
#         self.width = width
#     def forward(self, input):
#         input = f.interpolate(input, (self.height, self.width), mode="bilinear")
#         return input


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.shape)
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


# Simple Encoder
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.convolve = nn.Sequential(
            nn.Conv2d(3, 27, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.Conv2d(27, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.Conv2d(27, 81, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(81),
            nn.ELU(),
            nn.Conv2d(81, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            # nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(247),
            # nn.ELU(),
            # nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(247),
            # nn.ELU(),
            # End convolutions, start linear
            Flatten(),
            nn.Linear(hidden_dim, 8192),
            nn.ELU(),
            # nn.Linear(2048, 1024),
            # nn.ELU(),
            # nn.Linear(1024, 512)
        )

        # hidden => mu
        self.fc_mean = nn.Linear(8192, latent_dim)

        # hidden => logvar
        self.fc_var = nn.Linear(8192, latent_dim)

        # self.training = True

    # Encoder needs to produce mean and log variance (parameters of data distriubution)
    def forward(self, x):
        h_ = self.convolve(x)
        mean = self.fc_mean(h_)
        log_var = self.fc_var(h_)

        return mean, log_var


# Simple Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.deconvolve = nn.Sequential(
            # nn.Linear(latent_dim, 512),
            # nn.ELU(),
            # nn.Linear(512, 1024),
            # nn.ELU(),
            nn.Linear(latent_dim, 8192),
            nn.ELU(),
            nn.Linear(8192, hidden_dim),
            nn.ELU(),
            # End linear, Start deconvolve
            Unflatten(247, 7, 7),
            # nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(247),
            # nn.ELU(),
            # nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(247),
            # nn.ELU(),
            nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.ConvTranspose2d(247, 81, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(81),
            nn.ELU(),
            nn.ConvTranspose2d(81, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.ConvTranspose2d(27, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.ConvTranspose2d(27, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(3),
        )

    def forward(self, x):
        x_hat = self.deconvolve(x)
        x_hat = F.interpolate(
            x_hat, size=(200, 200), mode="bilinear", align_corners=True
        )
        x_hat = self.tanh(x_hat)
        return x_hat


# Model of the VAE
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)  # Sampling epsilon
        z = mean + var * epsilon  # Reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # Takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

loss1 = torch.nn.MSELoss(reduction="mean")


# Define loss and optimizer
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = loss1(x_hat, x)
    kld = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    return (loss_factor * reproduction_loss) + (kld)


optimizer = Adam(model.parameters(), lr=lr)

# Training

print("Start training VAE...")
model.train()
loss_list = []

smallest_loss = 10**9

for epoch in range(epochs):
    total_loss = 0
    count = 0
    for idx, data in enumerate(tqdm(train_loader)):
        if idx >= 20:
            break

        images, annotations = data
        images = images.to(DEVICE)
        panel = annotations["solar_panel"][0]
        if panel == 1:
            continue

        recon_images, mu, logvar = model(images)
        loss = loss_function(images, recon_images, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        count += 1
        optimizer.step()
        optimizer.zero_grad()

    print(f"Done with epoch: {epoch+1}! Average train loss is {total_loss/count}")
    if (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
        print("Saving Model...")
        torch.save(model, f"saved_models/vae/epoch_{epoch+1}.pt")
        print("Model Saved!")
    if (total_loss / count) < smallest_loss:
        torch.save(model, "saved_models/vae/best_epoch.pt")
        smallest_loss = total_loss / count
        print("Saved best model by train loss")
