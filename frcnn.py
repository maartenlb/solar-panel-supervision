import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SolarObject, collate_fn  # import your custom dataset

num_epochs = 10
seed = 42
batch_size = 5
lr = 0.0001

# Seed everything
torch.manual_seed(seed)
# np.random.seed(seed)

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
# Define the Faster R-CNN model with a ResNet-50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # number of object classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set the model to the device
model.to(device)

# Define the loss function and optimizer
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(num_epochs):
    print(f"Training Epoch: {epoch+1}!")
    counter = 0
    total_loss = 0
    # Train for one epoch
    for images, annotations in tqdm(train_loader):
        images = list(image.to(device) for image in images)

        targets = []
        for i in range(len(images)):
            d = {}
            d["boxes"] = annotations["boxes"][i]
            d["labels"] = annotations["labels"][i]
            targets.append(d)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute the classification and bounding box regression losses
        loss_dict = model(images, targets)
        cls_loss = loss_dict["loss_classifier"]
        bbox_loss = loss_dict["loss_box_reg"]
        loss = cls_loss + bbox_loss

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1

    # Print the loss and accuracy for each epoch
    print(
        f"Epoch {epoch+1} Train Loss: {total_loss/(counter * batch_size)} Val Loss: {0}"
    )
    if (epoch + 1) % 1 == 0 or epoch + 1 == num_epochs:
        print("Saving Model...")
        torch.save(model, f"saved_models/frcnn/epoch_{epoch+1}.pt")
        print("Model Saved!")
