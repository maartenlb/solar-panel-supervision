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
import matplotlib.pyplot as plt

# Define the hyperparameters
seed = 42
batch_size = 1

# Seed everything
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

model = torch.load("saved_models/vae/best_epoch.pt", map_location=device)
recon_loss = torch.nn.MSELoss(reduction="mean")


recon_loss = torch.nn.MSELoss(reduction="mean")


def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.float32(image.squeeze().transpose((1, 2, 0)))
    image = image[:, :, :3]

    return image


total_loss = 0
count = 0
for idx, data in enumerate(tqdm(train_loader)):
    if idx >= 20:
        break

    images, annotations = data
    images = images.to(device)
    panel = annotations["solar_panel"][0]

    recon_images, _, _ = model(images)
    loss = recon_loss(images, recon_images)
    total_loss += loss.item()
    count += 1

    recon_images = tensor_to_image(recon_images)
    images = tensor_to_image(images)

    if panel == 1:
        # Plot the image
        plt.imshow(recon_images)
        plt.axis("off")
        plt.savefig(
            f"output/vae_recon/solar_panel/img_{idx}_recon.png",
            bbox_inches="tight",
        )
        plt.clf()

        plt.imshow(images)
        plt.axis("off")
        plt.savefig(
            f"output/vae_recon/solar_panel/img_{idx}_original.png",
            bbox_inches="tight",
        )
        plt.clf()

        print(f"Reconstructed solar panel, loss is {loss}")
    else:
        # Plot the image
        plt.imshow(recon_images)
        plt.axis("off")
        plt.savefig(
            f"output/vae_recon/no_panel/img_{idx}_recon.png",
            bbox_inches="tight",
        )
        plt.clf()

        plt.imshow(images)
        plt.axis("off")
        plt.savefig(
            f"output/vae_recon/no_panel/img_{idx}_original.png",
            bbox_inches="tight",
        )
        plt.clf()
        print(f"Reconstructed without panel, loss is {loss}")
print(f"Done, used {count} images.")
