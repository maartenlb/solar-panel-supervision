import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
from dataset import SolarObject, collate_fn
from captum.attr import IntegratedGradients
from metrics import calc_metrics

# import torch.nn.functional as f
import matplotlib.pyplot as plt
import os.path
import json
from PIL import Image, ImageDraw


from pytorch_grad_cam import GradCAM, EigenCAM, FullGrad, LayerCAM

# Model Hyperparameters

epochs = 5000

seed = 42
batch_size = 1
lr = 0.000005
loss_factor = 300  # weight of recon loss between 0 and 1

resize = 200

latent_dim = 4096
hidden_dim = 247 * 7 * 7

# Seed everything
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define the device to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

data_transforms = transforms.Compose(
    [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5051, 0.5083, 0.4524], std=[0.1720, 0.1332, 0.1220]
        ),
    ]
)


# Load your custom dataset
dataset = SolarObject(
    data_dir="data/processed_imgs/",
    json_path="data/image_polygons.json",
    transform=data_transforms,
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


# Help functions
class Pleaseprint(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.shape)
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


# Simple Encoder
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.convolve = nn.Sequential(
            nn.Conv2d(3, 27, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.Conv2d(27, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.Conv2d(27, 81, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(81),
            nn.ELU(),
            nn.Conv2d(81, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.Conv2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.Conv2d(247, 247, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            # End convolutions, start linear
            Flatten(),
            nn.Linear(hidden_dim, 8192),
            nn.ELU(),
            nn.Linear(8192, 6144),
            nn.ELU(),
            # nn.Linear(2048, 1024),
            # nn.ELU(),
            # nn.Linear(1024, 512)
        )

        # hidden => mu
        self.fc_mean = nn.Linear(6144, latent_dim)

        # hidden => logvar
        self.fc_var = nn.Linear(6144, latent_dim)

        # self.training = True

    # Encoder needs to produce mean and log variance (parameters of data distriubution)
    def forward(self, x):
        h_ = self.convolve(x)
        mean = self.fc_mean(h_)
        log_var = self.fc_var(h_)

        return mean, log_var


# Simple Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Decoder, self).__init__()

        self.deconvolve = nn.Sequential(
            # nn.Linear(latent_dim, 512),
            # nn.ELU(),
            # nn.Linear(512, 1024),
            # nn.ELU(),
            nn.Linear(latent_dim, 6144),
            nn.ELU(),
            nn.Linear(6144, 8192),
            nn.ELU(),
            nn.Linear(8192, hidden_dim),
            nn.ELU(),
            # End linear, Start deconvolve
            Unflatten(247, 7, 7),
            nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.ConvTranspose2d(247, 247, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(247),
            nn.ELU(),
            nn.ConvTranspose2d(247, 81, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(81),
            nn.ELU(),
            nn.ConvTranspose2d(81, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.ConvTranspose2d(27, 27, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(27),
            nn.ELU(),
            nn.ConvTranspose2d(27, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, x):
        x_hat = self.deconvolve(x)
        x_hat = nn.functional.interpolate(
            x_hat, size=(200, 200), mode="bilinear", align_corners=True
        )
        return x_hat


# Model of the VAE
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)  # Sampling epsilon
        z = mean + var * epsilon  # Reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # Takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

loss1 = torch.nn.MSELoss(reduction="mean")


# Define loss and optimizer
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = loss1(x_hat, x)
    kld = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    return (loss_factor * reproduction_loss) + (kld)


optimizer = Adam(model.parameters(), lr=lr)

# Training
if os.path.exists(f"saved_models/vae/epoch_{epochs}.pt"):
    print("Found saved model, skipping training...")
    model = torch.load(f"saved_models/vae/best_epoch.pt", map_location=DEVICE)
else:
    print("Start training VAE...")
    model.train()
    smallest_loss = 10**10

    for epoch in range(epochs):
        overall_loss = 0
        counter = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 1000:
                break
            panel = y["solar_panel"][0]
            if panel == 1:
                continue
            x = x.to(DEVICE)

            x_hat, mean, log_var = model(x)

            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            # print(loss.item())

            counter += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = overall_loss / (counter * batch_size)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", avg_loss)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print("Saving Model...")
            torch.save(model, f"saved_models/vae/epoch_{epoch+1}.pt")
            print("Model Saved!")
        if avg_loss < smallest_loss:
            smallest_loss = avg_loss
            print("New best model found, saving...")
            torch.save(model, "saved_models/vae/best_epoch.pt")
            print("Model Saved!")
    print("Finish!")

    # # s = json.dumps(loss_list)
    # print("Writing loss to json...")
    # with open(
    #     f"saved_models/{dataset}/loss_{dataset}_epoch_{epoch+1}.json", "w"
    # ) as loc:
    #     json.dump(loss_list, loc)
    # print("Succes! Loss numbers saved in .json")


invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.1720, 1 / 0.1332, 1 / 0.1220]
        ),
        transforms.Normalize(mean=[-0.5051, -0.5083, -0.4524], std=[1.0, 1.0, 1.0]),
    ]
)


model.eval()
savedir = "output/vae_recon/"


class VAE_wrapper(torch.nn.Module):
    def __init__(self, model):
        super(VAE_wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x, mean, log_var = self.model(x)
        x = x.view(1, 120000)
        return x


target_model = VAE_wrapper(model=model)

target_layer = [
    # target_model.model.Encoder.convolve[0],
    # target_model.model.Encoder.convolve[3],
    # target_model.model.Encoder.convolve[6],
    # target_model.model.Encoder.convolve[9],
    # target_model.model.Encoder.convolve[12],
    # target_model.model.Encoder.convolve[15],
    # target_model.model.Encoder.convolve[18],
    # target_model.model.Decoder.deconvolve[10],
    # target_model.model.Decoder.deconvolve[13],
    # target_model.model.Decoder.deconvolve[16],
    target_model.model.Decoder.deconvolve[19],
    target_model.model.Decoder.deconvolve[22],
    target_model.model.Decoder.deconvolve[25],
]

save_img = True

cam = GradCAM(model=target_model, target_layers=target_layer)

cam_threshold = 0.3


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
for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
    if batch_idx >= 1000:
        break
    x = x.to(DEVICE)
    # model.zero_grad()
    panel = y["solar_panel"][0]

    if panel == 0:
        continue

    x_hat, mean, log_var = model(x)
    loss = loss1(x_hat, x)
    loss.backward()

    # get the Grad-CAM activation map
    cam_mask = cam(x)

    x = x.squeeze()
    x_hat = x_hat.squeeze()
    x_hat = invTrans(x_hat)
    x = invTrans(x)

    # normalize the CAM mask
    cam_mask = cam_mask - np.min(cam_mask)
    cam_mask = cam_mask / np.max(cam_mask)

    cam_mask = np.abs(np.squeeze(np.float32(cam_mask)))

    kernel = np.array(
        [0.125, 0.25, 0.5, 0.25, 0.125]
    )  # Apply Gaussian blur to gradient visualization
    cam_mask = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"), 0, cam_mask
    )
    cam_mask = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"), 1, cam_mask
    )

    if panel == 1:
        pred_mask = torch.from_numpy(cam_mask)
        pred_mask = torch.where(pred_mask >= cam_threshold, 1, 0).int()

        box_mask = torch.zeros((200, 200))
        bboxes = y["boxes"][0].int()
        for box in bboxes[1:]:
            x1, y1, x2, y2 = box
            box_mask[y1:y2, x1:x2] = 1

        # Create a new mask with the same shape as the image
        polygons = y["polygons"][0]
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

        avg_dict["bbox_iou"] = (count * avg_dict["bbox_iou"] + bbox_iou) / (count + 1)
        avg_dict["bbox_dice"] = (count * avg_dict["bbox_dice"] + bbox_dice) / (
            count + 1
        )
        avg_dict["bbox_prec"] = (count * avg_dict["bbox_prec"] + bbox_precision) / (
            count + 1
        )
        avg_dict["bbox_rec"] = (count * avg_dict["bbox_rec"] + bbox_recall) / (
            count + 1
        )

        avg_dict["poly_iou"] = (count * avg_dict["poly_iou"] + poly_iou) / (count + 1)
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

        if save_img:
            if panel == 1:
                print(f"Solar panel found with: {loss.item()}")
                save_image(x_hat, f"{savedir}panel/img_recon_{batch_idx}.png")
                save_image(x, f"{savedir}panel/img_orig_{batch_idx}.png")
            if panel == 0:
                print(f"No panel with {loss.item()}")
                save_image(x_hat, f"{savedir}no_panel/img_recon_{batch_idx}.png")
                save_image(x, f"{savedir}no_panel/img_orig_{batch_idx}.png")
            # convert the original image to PIL Image

            image = x.cpu().detach().numpy()
            image = np.float32(image.transpose((1, 2, 0)))
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
            if panel == 1:
                plt.savefig(
                    f"{savedir}cam/solar_panel_{batch_idx}.png",
                    bbox_inches="tight",
                )
            if panel == 0:
                plt.savefig(
                    f"{savedir}cam/no_panel_{batch_idx}.png",
                    bbox_inches="tight",
                )
            plt.clf()


with open(
    f"output/vae_localization/vae_cam_{cam_threshold}_avg_dict.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(avg_dict, f, ensure_ascii=False, indent=4)


#     loss.backward()

#     print(f"input shape is {x.shape}")
#     print(f"biggest grad is {torch.max(x)}")

#     print(f"biggest grad is {torch.min(x)}")

#     print(f"output shape is {x.shape}")
#     # x_ = model.Encoder.convolve[0](x)
#     # print(x_.shape)
#     # x_ = model.Encoder.convolve[1](x_)
#     # x_ = model.Encoder.convolve[2](x_)
#     # print(x_.shape)

#     grads = model.Encoder.convolve[3].weight.grad
#     grads = torch.abs(grads)
#     print(f"biggest grad is {torch.max(grads)}")
#     print(f"grads shape is {grads.shape}")

#     # grads = torch.mean(grads, 0)
#     # grads = torch.mean(grads, 0)

#     print(f"after mean grads shape is {grads.shape}")

#     grads = torch.nn.functional.interpolate(grads, (resize, resize), mode="bilinear")
#     grads = torch.mean(grads, 1)
#     grads = torch.mean(grads, 0)
#     grads = grads.resize_(1, resize, resize)

#     print(f"after interpolate grads shape is {grads.shape}")
#     print(f"after interpolate biggest grad is {torch.max(grads)}")

#     # grads = torch.mean(grads, 0)
#     # print(torch.max(grads))
#     # print(grads.shape)
#     # grads = torch.mean(grads, 0)
#     # print(torch.max(grads))
#     # print(grads.shape)
#     # print(grads)

#     # grads = model.Decoder.deconvolve[2].weight.grad
#     # grads = torch.abs(grads)
#     # grads = torch.mean(grads, 1)
#     # grads = torch.unsqueeze(grads, dim=0)
#     # grads = model.Decoder.deconvolve[4](grads)
#     # grads = model.Decoder.deconvolve[6](grads)
#     # grads = model.Decoder.deconvolve[8](grads)
#     # print(torch.max(grads))
#     # grads = torch.nn.functional.interpolate(grads, (28, 28), mode='bilinear')

#     # grads = torch.abs(grads)
#     # grads = torch.mean(grads, 0)
#     # grads = torch.sum(grads, 0)
#     # print(torch.max(grads))
#     # print(grads.shape)

#     x_image = x.view(3, resize, resize)
#     grads_image = grads.view(1, resize, resize)
#     x_hat_image = x_hat.view(3, resize, resize)

#     save_image(x_image, f"{savedir}{batch_idx}_{y.item()}_base.png")
#     save_image(grads_image, f"{savedir}{batch_idx}_{y.item()}_grads.png")
#     save_image(x_hat_image, f"{savedir}{batch_idx}_{y.item()}_encode.png")

#     x_image = (
#         x_image.view(x_image.shape[1], x_image.shape[2], x_image.shape[0])
#         .cpu()
#         .detach()
#         .numpy()
#     )
#     grads_image = (
#         grads_image.view(
#             grads_image.shape[1], grads_image.shape[2], grads_image.shape[0]
#         )
#         .cpu()
#         .detach()
#         .numpy()
#     )

#     # print(x_image.shape)
#     # print(grads_image.shape)

#     kernel = np.array(
#         [0.5, 1.0, 2.0, 1.0, 0.5]
#     )  # Apply Gaussian blur to gradient visualization
#     grads_image = np.apply_along_axis(
#         lambda x: np.convolve(x, kernel, mode="same"), 0, grads_image
#     )
#     grads_image = np.apply_along_axis(
#         lambda x: np.convolve(x, kernel, mode="same"), 1, grads_image
#     )

#     plt.imshow(x_image, cmap="gray", alpha=1)
#     plt.imshow(grads_image, cmap="jet", alpha=0.5)
#     plt.clim(0, 2)
#     plt.savefig(f"{savedir}{batch_idx}_{y.item()}_superimpose.png")
#     plt.clf()

#     if batch_idx >= 30:
#         break
