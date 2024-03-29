import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter
from pprint import pprint
from dataset import SolarObject, collate_fn  # import your custom dataset

num_epochs = 3
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


data_usage = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for data_chance in data_usage:
    # Define the Faster R-CNN model with a ResNet-50 backbone
    num_classes = 2  # number of object classes
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        num_classes=num_classes, trainable_backbone_layers=5
    )

    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set the model to the device
    model.to(device)
    model.train()

    # Define the loss function and optimizer
    weight = torch.tensor([1.0, 20.0])
    cls_criterion = nn.CrossEntropyLoss(weight=weight.to(device))
    bbox_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch+1}!")
        counter = 0
        total_loss = 0
        # Train for one epochf (epoch + 1) % 5 == 0 or
        for images, annotations in tqdm(train_loader):
            number = random.random()
            if number >= data_chance:
                continue

            images = list(image.to(device) for image in images)

            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations["boxes"][i].to(device)
                d["labels"] = annotations["labels"][i].to(device)
                targets.append(d)

            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute the classification and bounding box regression losses
            loss_dict = model(images, targets)
            cls_loss = loss_dict["loss_classifier"]
            bbox_loss = loss_dict["loss_box_reg"]
            loss = cls_loss + bbox_loss

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            counter += 1
            optimizer.zero_grad()

        # Print the loss and accuracy for each epoch
        print(
            f"Epoch {epoch+1} Train Loss: {total_loss/(counter * batch_size)} Val Loss: {0}"
        )
        if epoch + 1 == num_epochs:
            print("Saving Model...")
            torch.save(
                model,
                f"data_decrease/frcnn/epoch_{epoch+1}_data_usage_{data_chance*100}.pt",
            )
            print("Model Saved!")
