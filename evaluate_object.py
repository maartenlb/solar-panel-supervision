import torch
import numpy as np
import torchvision
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn  # import your custom dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate(model, data_loader, iou_threshold=0.5, dice_threshold=0.5):
    model.train()
    metric = MeanAveragePrecision(
        num_classes=3, iou_thresholds=[0.5, 0.75, 0.90], class_metrics=True
    )
    metric.eval()
    counter = 0
    for images, annotations in tqdm(data_loader):
        targets = []
        for i in range(len(images)):
            d = {}
            d["boxes"] = annotations["boxes"][i].to(device)
            d["labels"] = annotations["labels"][i].to(device)
            targets.append(d)
        preds = model(images, targets)
        pprint(preds)

        metric.update(preds, targets)
        counter += 1
        if counter > 20:
            break
    pprint(metric.compute())


transforms = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
seed = 42
batch_size = 1

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

model = torch.load("saved_models/frcnn/epoch_3.pt", map_location=device)

model.train()
model.to(device)

evaluate(model=model, data_loader=test_loader)
