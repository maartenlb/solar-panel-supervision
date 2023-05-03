import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    Initializes a basic block for the ResNet model

    Args:
        inplanes (int): The number of input channels.
        planes (int): The number of output channels.
        stride (int): The stride of the convolutional layer. Default: 1.
        downsample (nn.Module): A downsampling module. Default: None.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    class Encoder(nn.Module):
        """
        Encoder class for a ResNet-based Variational Autoencoder (VAE).

        Args:
            block (nn.Module): Block type to use for layers in ResNet.
            layers (list): List of layer counts for each of the four stages of ResNet.
            latent_dim (int): Dimensionality of the latent space.
        """

    def __init__(self, block, layers, latent_dim):
        super(Encoder, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Helper function to create a layer in the ResNet encoder.

        Args:
            block (nn.Module): Block type to use for layer in ResNet.
            planes (int): Number of output channels for layer.
            blocks (int): Number of blocks to use for layer.
            stride (int): Stride for layer (default=1).

        Returns:
            nn.Sequential: Sequence of blocks for the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class DecoderBasicBlock(nn.Module):
    """
    Initializes a DecoderBasicBlock object.
    We use a separate class since the decoder block needs to deconvolve rather than convolve.

    Args:
    - inplanes (int): the number of input channels
    - planes (int): the number of output channels
    - stride (int): the stride used in the first convolutional layer. Default: 1
    - upsample (nn.Module): a module that performs upsampling on the input tensor if necessary.
        Default: None
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(DecoderBasicBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, block, layers, latent_dim):
        super(Decoder, self).__init__()

        self.inplanes = 512 * block.expansion
        self.latent_dim = latent_dim
        self.layers = layers

        # First linear layer
        self.fc = nn.Linear(self.latent_dim, 512 * 4 * 4)

        # Residual layers
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.layer4 = self._make_layer(block, 64, layers[3])

        # Transposed convolutions to upsample
        self.deconv1 = nn.ConvTranspose2d(
            64 * block.expansion,
            64 * block.expansion,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(64 * block.expansion)
        self.deconv2 = nn.ConvTranspose2d(
            64 * block.expansion,
            32 * block.expansion,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(32 * block.expansion)
        self.deconv3 = nn.ConvTranspose2d(
            32 * block.expansion,
            16 * block.expansion,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(16 * block.expansion)
        self.deconv4 = nn.ConvTranspose2d(
            16 * block.expansion, 3, kernel_size=4, stride=2, padding=1
        )

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))
        return x


class VAE(nn.Module):
    def __init__(self, block, layers, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(block, layers, latent_dim)
        self.decoder = Decoder(block, layers, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
