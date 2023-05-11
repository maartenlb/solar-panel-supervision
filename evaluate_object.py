import torch
import numpy as np
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn  # import your custom dataset
from metrics import calc_metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import json


def evaluate(model, data_loader, threshold):
    model.eval()
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
    pred_labels = []
    true_labels = []
    for images, annotations in tqdm(data_loader):
        images = images.to(device)
        targets = []
        for i in range(len(images)):
            d = {}
            d["boxes"] = annotations["boxes"][i].to(device)
            d["labels"] = annotations["labels"][i].to(device)
            targets.append(d)
        preds = model(images)

        pred_panel = False
        for s in preds[0]["scores"]:
            if s.item() >= threshold:
                pred_panel = True

        if pred_panel:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        if len(annotations["labels"][0]) > 1:
            true_labels.append(1)
        else:
            true_labels.append(0)

        if (pred_labels[-1] == 1) and (true_labels[-1] == 1):
            pred_mask = torch.zeros((200, 200))

            valid_boxes = preds[0]["scores"] >= threshold
            box_tensor = preds[0]["boxes"][valid_boxes].int()

            for box in box_tensor:
                x1, y1, x2, y2 = box
                pred_mask[y1:y2, x1:x2] = 1

            # calculate localization metrics:
            box_mask = torch.zeros((200, 200))
            bboxes = annotations["boxes"][0].int()
            for box in bboxes[1:]:
                x1, y1, x2, y2 = box
                box_mask[y1:y2, x1:x2] = 1

            # Create a new mask with the same shape as the image
            polygons = annotations["polygons"][0]
            poly_img = Image.new("L", (200, 200), 0)
            for poly in polygons:
                poly = [item for sublist in poly for item in sublist]

                ImageDraw.Draw(poly_img).polygon(poly, outline=1, fill=1)

            poly_mask = np.array(poly_img)
            poly_mask = torch.from_numpy(poly_mask).int()

            bbox_iou, bbox_dice, bbox_precision, bbox_recall = calc_metrics(
                pred_mask, box_mask
            )
            poly_iou, poly_dice, poly_precision, poly_recall = calc_metrics(
                pred_mask, poly_mask
            )

            avg_dict["bbox_iou"] = (count * avg_dict["bbox_iou"] + bbox_iou) / (
                count + 1
            )
            avg_dict["bbox_dice"] = (count * avg_dict["bbox_dice"] + bbox_dice) / (
                count + 1
            )
            avg_dict["bbox_prec"] = (count * avg_dict["bbox_prec"] + bbox_precision) / (
                count + 1
            )
            avg_dict["bbox_rec"] = (count * avg_dict["bbox_rec"] + bbox_recall) / (
                count + 1
            )

            avg_dict["poly_iou"] = (count * avg_dict["poly_iou"] + poly_iou) / (
                count + 1
            )
            avg_dict["poly_dice"] = (count * avg_dict["poly_dice"] + poly_dice) / (
                count + 1
            )
            avg_dict["poly_prec"] = (count * avg_dict["poly_prec"] + poly_precision) / (
                count + 1
            )
            avg_dict["poly_rec"] = (count * avg_dict["poly_rec"] + poly_recall) / (
                count + 1
            )

            count += 1

            if poly_iou > 0:
                avg_dict["correct"] += 1
            else:
                avg_dict["incorrect"] += 1
    acc = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1score = f1_score(true_labels, pred_labels)
    target_names = ["background", "solar_panel"]

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1score}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print(
        classification_report(
            true_labels, pred_labels, target_names=target_names, digits=4
        )
    )

    cm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    print(f"Accuracy by class:{cm.diagonal()}")

    return avg_dict


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

model = torch.load("saved_models/frcnn/epoch_10.pt", map_location=device)

model.to(device)
threshold_list = [0.001, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.999]
for threshold in threshold_list:
    print(f"Running with Threshold: {threshold}")

    avg_dict = evaluate(model=model, data_loader=test_loader, threshold=threshold)

    with open(
        f"output/object_detection/threshold_{threshold}_avg_dict.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(avg_dict, f, ensure_ascii=False, indent=4)
