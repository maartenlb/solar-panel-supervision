import os
import json
from PIL import Image
import torch
import numpy as np
import torch.utils.data as torchdata
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SolarObject(Dataset):
    def __init__(self, data_dir, json_path, transform=None, random_seed=42):

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.data_dir = data_dir
        self.json_path = json_path
        self.transform = transform

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        self.image_names = list(self.data.keys())

        num_samples = len(self.image_names)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_split = int(0.8 * num_samples)
        val_split = int(0.1 * num_samples)
        self.train_indices = indices[:train_split]
        self.val_indices = indices[train_split : train_split + val_split]
        self.test_indices = indices[train_split + val_split :]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        boxes.append([0, 0, 199, 199])
        labels.append(0)
        solar_panel = False
        for box in self.data[image_name]["bounding_boxes"]:
            boxes.append(box)
            labels.append(1)  # Only one object class present
            solar_panel = True

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "solar_panel": solar_panel}

        if self.transform:
            image = self.transform(image)

        return image, target

    def train_sampler(self):
        return torchdata.sampler.SubsetRandomSampler(self.train_indices)

    def val_sampler(self):
        return torchdata.sampler.SubsetRandomSampler(self.val_indices)

    def test_sampler(self):
        return torchdata.sampler.SubsetRandomSampler(self.test_indices)


def collate_fn(batch):
    # Create lists to store the targets
    images = []
    boxes = []
    labels = []
    for image, target in batch:
        # Get the bounding boxes and labels for this image
        image_boxes = target["boxes"]
        image_labels = target["labels"]

        # Append the boxes and labels to the lists
        boxes.append(image_boxes)
        labels.append(image_labels)
        images.append(image)

    # Create padded tensors for the boxes and labels
    padded_boxes = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    # Remove boxes with zero height or width
    valid_boxes = []
    for boxes in padded_boxes:
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid_boxes.append(boxes[valid_mask])

    # Create a dictionary to hold the targets
    targets = {"boxes": valid_boxes, "labels": padded_labels}

    # Stack resized images into a batch tensor
    images = torch.stack(images, dim=0)

    return images, targets
