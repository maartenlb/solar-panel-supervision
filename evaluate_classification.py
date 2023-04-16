import torch
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import random
from PIL import Image
from torchvision.transforms import ToPILImage
from pprint import pprint
import matplotlib.pyplot as plt


def evaluate(model, dataloader, device):
    # Initialize the lists to store predictions and ground truth labels
    preds = []
    labels = []
    counter = 0

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
            preds.extend(torch.round(probs).cpu().numpy().astype(int).tolist())
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


def CAM_map(model, dataloader, cam, device):
    # pass the images through the model to get the output activations
    for idx, batch in enumerate(tqdm(dataloader)):
        images, annotations = batch
        targets = []
        for j in range(len(images)):
            targets.append(annotations["solar_panel"][j])
        targets = torch.as_tensor(targets, dtype=torch.float32).to(device)

        outputs = model(images)

        # get the predicted class for each image
        preds = torch.round(torch.sigmoid(outputs)).squeeze()

        for i in range(len(images)):
            if preds[i] == 1:
                # get the Grad-CAM activation map for the predicted class
                cam_mask = cam(images[i].unsqueeze(0))
                image = images[i].cpu().detach().numpy()
                # apply the mask to the original image to get the Grad-CAM image
                # gradcam_image = mask * images[i].cpu().detach().numpy()

                # normalize the CAM mask
                cam_mask = cam_mask - np.min(cam_mask)
                cam_mask = cam_mask / np.max(cam_mask)

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
                        f"output/cam/tp/solar_panel_{idx}_{i}.png", bbox_inches="tight"
                    )
                if targets[i] == 0:
                    plt.savefig(
                        f"output/cam/fp/solar_panel_{idx}_{i}.png", bbox_inches="tight"
                    )
                plt.clf()


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

model = torch.load("saved_models/classifier/resnet50/epoch_10.pt", map_location=device)

model.to(device)
model.eval()
target_layer = model.layer4

cam = GradCAM(model=model, target_layers=target_layer)

CAM_map(model=model, dataloader=test_loader, cam=cam, device=device)

evaluate(model=model, dataloader=test_loader, device=device)
