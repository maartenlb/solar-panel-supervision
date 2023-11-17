import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import SolarObject, collate_fn  # import your custom dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from pytorch_grad_cam import (
    GradCAM,
    EigenCAM,
    EigenGradCAM,
    GradCAMPlusPlus,
    FullGrad,
    HiResCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import random
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
from pprint import pprint
import matplotlib.pyplot as plt
from metrics import calc_metrics
from pprint import pprint
import json


def evaluate(model, dataloader, threshold, device):
    # Initialize the lists to store predictions and ground truth labels
    preds = []
    labels = []

    # Iterate over the validation dataset
    with torch.no_grad():
        for inputs, annotations in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = []
            for j in range(len(inputs)):
                targets.append(annotations["solar_panel"][j])
            targets = torch.as_tensor(targets, dtype=torch.float32).to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            probs = threshold(probs)
            probs = torch.round_(probs)

            preds.extend(probs.cpu().numpy().astype(int).tolist())
            labels.extend(targets.cpu().numpy().astype(int).tolist())

    # Calculate accuracy and confusion matrix
    acc = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1score = f1_score(labels, preds)
    target_names = ["background", "solar_panel"]

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1score}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    cm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    print(f"Accuracy by class:{cm.diagonal()}")


def CAM_map(model, dataloader, threshold, cam, cam_threshold, device, save_img=True):
    # pass the images through the model to get the output activations
    avg_dict = {
        "bbox_iou": 0,
        "bbox_dice": 0,
        "bbox_prec": 0,
        "bbox_rec": 0,
        "poly_iou": 0,
        "poly_dice": 0,
        "poly_prec": 0,
        "poly_rec": 0,
        "correct": 0,
        "incorrect": 0,
    }
    count = 0
    solar_count = 0
    no_solar = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        images, annotations = batch
        targets = []
        for j in range(len(images)):
            targets.append(annotations["solar_panel"][j])
        targets = torch.as_tensor(targets, dtype=torch.float32).to(device)

        for i in range(len(images)):
            if targets[i] == 1:
                solar_count += len(annotations["polygons"])
            else:
                no_solar += 1
    print(f"total solar panels in trainset: {solar_count}")
    print(f"total nothing in trainset: {no_solar}")
    return avg_dict


# Load the saved model
transforms = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
seed = 42
batch_size = 1

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = SolarObject(
    data_dir="data/processed_imgs/",
    json_path="data/image_polygons.json",
    transform=transforms,
    random_seed=seed,
)

test_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=dataset.test_sampler(),
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=dataset.train_sampler(),
)

model = torch.load("saved_models/classifier/resnet50/epoch_10.pt", map_location=device)

model.to(device)
model.eval()
target_layer = model.layer4

class_threshold_list = [0.9]
cam_threshold_list = [0.9]

cam = GradCAM(model=model, target_layers=target_layer)
print("Using GradCAM!!!")

for class_threshold in class_threshold_list:
    print(f"Classification Threshold for this run is: {class_threshold}")

    threshold = nn.Threshold(class_threshold, 0)
    # evaluate(model=model, dataloader=test_loader, threshold=threshold, device=device)

    for cam_threshold in cam_threshold_list:
        print(f"Camming with threshold {cam_threshold}")
        avg_dict = CAM_map(
            model=model,
            dataloader=train_loader,
            threshold=threshold,
            cam_threshold=cam_threshold,
            cam=cam,
            device=device,
        )
