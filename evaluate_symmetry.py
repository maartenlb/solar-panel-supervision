import torch
import numpy as np
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn
import json
import matplotlib.pyplot as plt


def evaluate(
    model_object,
    model_classifier,
    model_vae,
    data_loader,
    object_threshold,
    classification_threshold,
):
    model_object.eval()
    model_classifier.eval()
    symmetry_counter = 0
    asymmetry_counter = 0
    count = 0
    for idx, data in enumerate(tqdm(data_loader)):
        images, annotations = data
        images = images.to(device)
        targets = []
        for i in range(len(images)):
            d = {}
            d["boxes"] = annotations["boxes"][i].to(device)
            d["labels"] = annotations["labels"][i].to(device)
            d["panel"] = annotations["solar_panel"][i].to(device)
            targets.append(d)
        pred_object = model_object(images)

        object_pred_panel = False
        for s in pred_object[0]["scores"]:
            if s.item() >= object_threshold:
                object_pred_panel = True

        output_classifier = model_classifier(images)
        class_pred = torch.sigmoid(output_classifier).item()

        class_pred_panel = False
        if class_pred >= clasify_threshold:
            class_pred_panel = True

        if object_pred_panel != class_pred_panel:
            target_panel = False
            if targets[0] == 1:
                target_panel = True
            asymmetry_counter += 1
            image = images[i].cpu().detach().numpy()

            image = np.float32(image.squeeze().transpose((1, 2, 0)))
            image = image[:, :, :3]

            # Plot the blended image
            plt.imshow(image)
            plt.axis("off")
            if target_panel == object_pred_panel:
                plt.savefig(
                    f"output/symmetry/object_right/solar_panel_{idx}_{target_panel}.png",
                    bbox_inches="tight",
                )
            if target_panel == class_pred_panel:
                plt.savefig(
                    f"output/symmetry/classifier_right/solar_panel_{idx}_{target_panel}.png",
                    bbox_inches="tight",
                )
            plt.clf()
        else:
            symmetry_counter += 1

    print(f"Total symmetry: {symmetry_counter}")
    print(f"Total asymmetry: {asymmetry_counter}")


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

model_object = torch.load("saved_models/frcnn/epoch_10.pt", map_location=device)
model_classifier = torch.load(
    "saved_models/classifier/epoch_10.pt", map_location=device
)

model_object.to(device)
model_classifier.to(device)
object_threshold = 0.7
clasify_threshold = 0.9
print(
    f"Running with Object Threshold: {object_threshold} and Classification Threshold: {clasify_threshold}"
)

evaluate(
    model_object=model_object,
    model_classifier=model_classifier,
    model_vae=model_classifier,
    data_loader=test_loader,
    object_threshold=object_threshold,
    classification_threshold=clasify_threshold,
)
