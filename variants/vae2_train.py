import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn
from vae_model import VAE, Encoder, Decoder, DecoderBasicBlock, BasicBlock
from tqdm import tqdm

# Define the hyperparameters
num_epochs = 10000
seed = 42
batch_size = 1
lr = 0.00001
latent_dim = 8192
loss_factor = 1
# loss_factor = (200 * 200 * 3) / latent_dim  # Beta-VAE

# Seed everything
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose(
    [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5051, 0.5083, 0.4524], std=[0.1720, 0.1332, 0.1220]
        ),
    ]
)

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

# Define the model
model = VAE(
    encoder_block=BasicBlock,
    encoder_layers=[3, 4, 6, 3],
    decoder_block=DecoderBasicBlock,
    decoder_layers=[3, 6, 4, 3],
    latent_dim=latent_dim,
)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

recon_loss = torch.nn.MSELoss(reduction="mean")


def vae_loss(x_hat, x, mu, logvar):
    loss = recon_loss(x, x_hat)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld + loss_factor * loss


smallest_loss = 10**9
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for idx, data in enumerate(tqdm(train_loader)):
        if idx >= 20:
            break

        images, annotations = data
        images = images.to(device)
        panel = annotations["solar_panel"][0]
        if panel == 1:
            continue

        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        count += 1
        optimizer.step()
        optimizer.zero_grad()

    print(f"Done with epoch: {epoch+1}! Average train loss is {total_loss/count}")
    if (epoch + 1) % 1000 == 0 or epoch + 1 == num_epochs:
        print("Saving Model...")
        torch.save(model, f"saved_models/vae/epoch_{epoch+1}.pt")
        print("Model Saved!")
    if (total_loss / count) < smallest_loss:
        torch.save(model, "saved_models/vae/best_epoch.pt")
        smallest_loss = total_loss / count
        print("Saved best model by train loss")
