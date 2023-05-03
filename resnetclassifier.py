import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn
import numpy as np
import random

# Define the hyperparameters
num_epochs = 10
seed = 42
batch_size = 8
lr = 0.0001

# Seed everything
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])

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
model = resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # 2 classes: background and solar panel
model = model.to(device)


# Define the loss function and optimizer
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Calculate binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply class weights
        weights = targets * self.pos_weight + (1 - targets)
        loss = weights * loss

        # Average the loss across all samples
        loss = torch.mean(loss)

        return loss


criterion = WeightedBCELoss(pos_weight=20)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    counter = 0
    for data in tqdm(train_loader, 0):
        inputs, annotations = data
        inputs = inputs.to(device)
        targets = []
        for j in range(len(inputs)):
            targets.append(annotations["solar_panel"][j])
        targets = torch.as_tensor(targets, dtype=torch.float32).to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        counter += 1
        running_loss += loss.item()
    print(f"Done with epoch: {epoch+1}, Loss is {running_loss/(batch_size * counter)}")
    if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
        print("Saving Model...")
        torch.save(model, f"saved_models/classifier/epoch_{epoch+1}.pt")
        print("Model Saved!")
