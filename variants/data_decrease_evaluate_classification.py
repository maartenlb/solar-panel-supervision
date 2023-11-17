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
from pytorch_grad_cam import GradCAM, EigenCAM, EigenGradCAM, LayerCAM, GradCAMPlusPlus
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


def CAM_map(model, dataloader, threshold, cam, cam_threshold, device, save_img=False):
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
    for idx, batch in enumerate(tqdm(dataloader)):
        images, annotations = batch
        targets = []
        for j in range(len(images)):
            targets.append(annotations["solar_panel"][j])
        targets = torch.as_tensor(targets, dtype=torch.float32).to(device)

        images = images.to(device)

        outputs = model(images)

        # get the predicted class for each image
        preds = torch.sigmoid(outputs)
        preds = threshold(preds)
        preds = torch.round_(preds).squeeze()

        for i in range(len(images)):
            if preds[i] == 1:
                # get the Grad-CAM activation map for the predicted class
                cam_mask = cam(images[i].unsqueeze(0))
                # normalize the CAM mask
                cam_mask = cam_mask - np.min(cam_mask)
                cam_mask = cam_mask / np.max(cam_mask)

                if targets[i] == 1:
                    pred_mask = torch.from_numpy(cam_mask)
                    pred_mask = torch.where(pred_mask >= cam_threshold, 1, 0).int()
                    box_mask = torch.zeros((200, 200))
                    bboxes = annotations["boxes"][i].int()
                    for box in bboxes[1:]:
                        x1, y1, x2, y2 = box
                        box_mask[y1:y2, x1:x2] = 1

                    # Create a new mask with the same shape as the image
                    polygons = annotations["polygons"][i]
                    poly_img = Image.new("L", (200, 200), 0)
                    for poly in polygons:
                        poly = [item for sublist in poly for item in sublist]

                        ImageDraw.Draw(poly_img).polygon(poly, outline=1, fill=1)

                    poly_mask = np.array(poly_img)
                    poly_mask = torch.from_numpy(poly_mask).int()

                    # localization metrics
                    bbox_iou, bbox_dice, bbox_precision, bbox_recall = calc_metrics(
                        pred_mask, box_mask
                    )
                    poly_iou, poly_dice, poly_precision, poly_recall = calc_metrics(
                        pred_mask, poly_mask
                    )

                    avg_dict["bbox_iou"] = (count * avg_dict["bbox_iou"] + bbox_iou) / (
                        count + 1
                    )
                    avg_dict["bbox_dice"] = (
                        count * avg_dict["bbox_dice"] + bbox_dice
                    ) / (count + 1)
                    avg_dict["bbox_prec"] = (
                        count * avg_dict["bbox_prec"] + bbox_precision
                    ) / (count + 1)
                    avg_dict["bbox_rec"] = (
                        count * avg_dict["bbox_rec"] + bbox_recall
                    ) / (count + 1)

                    avg_dict["poly_iou"] = (count * avg_dict["poly_iou"] + poly_iou) / (
                        count + 1
                    )
                    avg_dict["poly_dice"] = (
                        count * avg_dict["poly_dice"] + poly_dice
                    ) / (count + 1)
                    avg_dict["poly_prec"] = (
                        count * avg_dict["poly_prec"] + poly_precision
                    ) / (count + 1)
                    avg_dict["poly_rec"] = (
                        count * avg_dict["poly_rec"] + poly_recall
                    ) / (count + 1)

                    count += 1

                    if poly_iou > 0:
                        avg_dict["correct"] += 1
                    else:
                        avg_dict["incorrect"] += 1

                if save_img:
                    image = images[i].cpu().detach().numpy()

                    # resize the CAM mask to match the size of the original image
                    # cam_mask = cv2.resize(cam_mask, (image.shape[2], image.shape[1]))

                    # convert the CAM mask to PIL Image
                    cam_mask = np.squeeze(np.float32(cam_mask))
                    # convert the original image to PIL Image

                    image = np.float32(image.squeeze().transpose((1, 2, 0)))
                    image = image[:, :, :3]

                    # Create a colormap for the heatmap
                    heatmap_colormap = plt.cm.get_cmap("jet")

                    # Apply the colormap to the normalized heatmap mask
                    heatmap = heatmap_colormap(cam_mask)
                    heatmap = heatmap[:, :, :3]

                    # Blend the heatmap with the original image using alpha blending
                    alpha = 0.2
                    blended = (alpha * heatmap) + ((1 - alpha) * image)

                    # Plot the blended image
                    plt.imshow(blended)
                    plt.axis("off")
                    if targets[i] == 1:
                        plt.savefig(
                            f"output/cam/eigengradcam/tp/solar_panel_{idx}_{i}.png",
                            bbox_inches="tight",
                        )
                    if targets[i] == 0:
                        plt.savefig(
                            f"output/cam/eigengradcam/fp/solar_panel_{idx}_{i}.png",
                            bbox_inches="tight",
                        )
                    plt.clf()
    return avg_dict


# Load the saved model
transforms = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])
seed = 42
batch_size = 5

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


class_threshold_list = [0.9]
cam_threshold_list = [0.001, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.999]

data_usage = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for data_chance in data_usage:
    print(f"Using model with data usage of {data_chance}")
    model = torch.load(
        f"data_decrease/classifier/epoch_10_data_usage{data_chance*100}.pt",
        map_location=device,
    )
    model.eval()
    model.to(device)
    target_layer = model.layer4
    cam = GradCAM(model=model, target_layers=target_layer)
    print("Using gradCAM++!!!")

    for class_threshold in class_threshold_list:
        print(f"Classification Threshold for this run is: {class_threshold}")
        threshold = nn.Threshold(class_threshold, 0)
        evaluate(
            model=model, dataloader=test_loader, threshold=threshold, device=device
        )

        for cam_threshold in cam_threshold_list:
            print(f"Camming with threshold {cam_threshold}")
            avg_dict = CAM_map(
                model=model,
                dataloader=test_loader,
                threshold=threshold,
                cam_threshold=cam_threshold,
                cam=cam,
                device=device,
            )

            with open(
                f"data_decrease/classification/gradcamplusplus_class_{class_threshold}_cam_{cam_threshold}_avg_dict.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(avg_dict, f, ensure_ascii=False, indent=4)
