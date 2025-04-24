import math

import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import Sequential as Seq
from typing import List, Tuple, Dict
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torchvision import transforms

from GVAE.ADVisionGNN.gcn_lib import Grapher, act_layer
from GVAE.ADVisionGNN.model.patchcore.utils import save_to_faiss, subsampling, corset_subsampling


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """
    Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv =nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels= in_dim,
                    out_channels=out_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_dim))

    def forward(self, x):
        x = self.conv(x)
        return x


class GraphEncoder(nn.Module):
    """Graph Encoder"""
    def __init__(self,
                 num_block: List,
                 hidden_channels: List,
                 reduce_ratios: List,
                 dpr: List,
                 k: int = 9,
                 conv: str = 'mr',
                 act: str = 'gelu',
                 norm: str = 'batch',
                 bias: bool = True,
                 stochastic: bool = False,
                 epsilon: float = 0.2,
                 ):
        super().__init__()
        self.n_blocks = sum(num_block)
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        self.blocks = nn.ModuleList([])
        HW = 224 // 4 * 224 // 4
        idx = 0
        for i in range(len(num_block)):
            if i > 0:
                self.blocks.append(Downsample(
                    hidden_channels[i - 1],
                    hidden_channels[i]
                ))
                for j in range(num_block[i]):
                    self.blocks += [
                        nn.Sequential(
                            Grapher(
                                hidden_channels[i],
                                num_knn[idx],
                                1,
                                conv,
                                act,
                                norm,
                                bias,
                                stochastic,
                                epsilon,
                                reduce_ratios[i],
                                n=HW,
                                drop_path=dpr[idx],
                                relative_pos=False
                            ),
                            FFN(
                                hidden_channels[i],
                                hidden_channels[i] * 4,
                                act=act,
                                drop_path=dpr[idx])
                        )
                    ]

    def forward(self, x):
        for i in range(len(self.blocks)):
            # print(f"Passed through the {i} block with feature shape", x.shape)
            x = self.blocks[i](x)

        return x


class CNNDecoder(nn.Module):
    """Normal CNN Decoder"""
    def __init__(self,
                 hidden_channels: List,
                 kernel_sizes: List,
                 list_strides: List,
                 paddings: List
                 ):
        super().__init__()
        # [1, 384, 7, 7]
        self.conv_layers = nn.ModuleList([])
        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels= hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=list_strides[i],
                    padding=paddings[i]
                ),
                nn.BatchNorm2d(hidden_channels[i + 1]),
                nn.ReLU()
            ))

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels[-1],
                out_channels=3,
                kernel_size=kernel_sizes[-1],
                padding=paddings[-1],
                stride=list_strides[-1]
            ),
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class GraphDecoder(nn.Module):
    """Graph Decoder"""
    def __init__(self,
                 num_block: List,
                 hidden_channels: List,
                 reduce_ratios: List,
                 dpr: List,
                 k: int = 9,
                 conv: str = 'mr',
                 act: str = 'gelu',
                 norm: str = 'batch',
                 bias: bool = True,
                 stochastic: bool = False,
                 epsilon: float = 0.2,
                 ):
        super().__init__()
        self.n_blocks = sum(num_block)
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        self.blocks = nn.ModuleList([])
        HW = 224 // 4 * 224 // 4
        idx = 0
        for i in range(len(num_block)):
            if i > 0:
                self.blocks.append(UpSample(
                    hidden_channels[i - 1],
                    hidden_channels[i]
                ))
                for j in range(num_block[i]):
                    self.blocks += [
                        nn.Sequential(
                            Grapher(
                                hidden_channels[i],
                                num_knn[idx],
                                1,
                                conv,
                                act,
                                norm,
                                bias,
                                stochastic,
                                epsilon,
                                reduce_ratios[i],
                                n=HW,
                                drop_path=dpr[idx],
                                relative_pos=False
                            ),
                            FFN(
                                hidden_channels[i],
                                hidden_channels[i] * 4,
                                act=act,
                                drop_path=dpr[idx])
                        )
                    ]

    def forward(self, x):
        for i in range(len(self.blocks)):
            # print(f"Passed through the {i} block with feature shape", x.shape)
            x = self.blocks[i](x)

        return x


class GraphVariationalAutoencoder(nn.Module):
    def __init__(self,
                 decoder:str = 'cnn',
                 mode: str = "train",
                 memory_bank_path: str = "../model/results/model/memory_bank.index",):
        """

        :param decoder: cnn or graph
        :param mode: train or test mode
        """
        super(GraphVariationalAutoencoder, self).__init__()

        self.memory_bank_path = memory_bank_path
        self.memory_bank = list()
        self.mode = mode

        blocks = [2, 2, 6, 2]
        blocks = [2, 4, 2]
        self.n_blocks = sum(blocks)
        k = 9
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        channels = [48, 96, 240, 384]
        channels = [48, 96, 240,]# number of channels of deep features
        channels = [96, 240, 384]
        HW = 224 // 4 * 224 // 4
        max_dilation = 49 // max(num_knn)
        conv = 'mr'
        act = 'gelu'
        norm = 'batch'
        bias = True
        stochastic = False
        dropout = 0.0
        epsilon = 0.2
        drop_path = 0.0
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))

        self.encoder = GraphEncoder(
            num_block=blocks,
            hidden_channels=channels,
            reduce_ratios=reduce_ratios,
            dpr=dpr
        )

        if decoder == "cnn":
            self.decoder = CNNDecoder(
                hidden_channels=[384, 240, 96, 48],
                kernel_sizes=[4, 4, 4, 4],
                list_strides=[2, 2, 2, 2],
                paddings=[1, 1, 1, 1]
            )
        else:
            self.decoder = CNNDecoder(
                hidden_channels=[384, 240, 96, 48],
                kernel_sizes=[4, 4, 4, 4],
                list_strides=[2, 2, 2, 2],
                paddings=[1, 1, 1, 1]
            )
        self.model_init()

        self.features = []
        def hook(module, inputs, outputs):
            self.features.append(outputs)

        self.encoder.blocks[4][1].fc2[1].register_forward_hook(hook)
        self.encoder.blocks[7][1].fc2[1].register_forward_hook(hook)

        self.avg = torch.nn.AvgPool2d(3, stride=1)

        self.y_score= list()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def fill_memory_bank(self):
        self.memory_bank = subsampling(self.memory_bank)
        save_to_faiss(
            self.memory_bank,
            self.memory_bank_path
        )

    def forward(self, inputs, is_saved: bool = False):
        segm_map = None
        embedding_layer = self.stem(inputs) + self.pos_embed
        B, C, H, W = embedding_layer.shape

        encoder_output = self.encoder(embedding_layer)

        self.features = []
        with torch.no_grad():
            _ = self.encoder(embedding_layer)
        fmap_size = self.features[0].shape[-2]
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)  # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T  # Create a column tensor

        if self.mode == "train" and is_saved:
            print("Saving")
            # sub_patch = subsampling(patch)
            sub_patch = corset_subsampling(patch)
            print(sub_patch.shape)
            save_to_faiss(sub_patch, path=self.memory_bank_path)
        elif self.mode == "test":
            memory_bank = load_faiss_to_tensor(self.memory_bank_path).cuda()
            print("Test:", patch.shape, memory_bank.shape)
            distances = torch.cdist(
                patch,
                memory_bank
            )
            print(distances.shape)
            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            s_star = torch.max(dist_score)
            segm_map = dist_score.view(B, 1, 28, 28)
            segm_map = F.interpolate(segm_map, size=(224, 224), mode='bilinear', align_corners=False)

        x_reconstruct = self.decoder(encoder_output)
        if self.mode == 'test' and segm_map is not None:
            # x_after = x_before + segm_map.cuda()
            # print("Final:", x_before.shape)
            return x_reconstruct, segm_map
        return x_reconstruct, encoder_output

def test_saving_feature():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model
    model = GraphVariationalAutoencoder(decoder='cnn', mode='train').to(device)

    # Create dummy input data (e.g., a batch of 8 images with 3 channels, 224x224)
    dummy_input = torch.randn(32, 3, 224, 224).to(device)

    # Set the model to training mode
    model.train()

    # Call the forward method with is_saved set to True
    output = model(dummy_input, is_saved=True)

    # Verify output shape
    print("Output shape:", output.shape)

    # Check if memory bank is filled and saved correctly
    if model.features:
        print("Memory bank is filled.")
    else:
        print("Memory bank is empty.")


def load_faiss_to_tensor(index_path):
    # Load the FAISS index
    print("got here")
    index = faiss.read_index(index_path)

    if index is not None:
        # Get the number of vectors in the index
        num_vectors = index.ntotal
        feature_dim = index.d  # Dimension of the feature vectors

        # Create a tensor to hold the vectors
        vectors = torch.empty((num_vectors, feature_dim), dtype=torch.float32)
        # Retrieve all vectors
        for i in range(num_vectors):
            vectors[i] = torch.from_numpy(index.reconstruct(i))
        print(vectors.shape)
        return vectors

def load_image(image_path):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)
    return image.cuda()


# Function to test the model and plot the segm_map
def test_and_plot_segmentation(model, image_path):
    # Load and preprocess the image
    image = load_image(image_path)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Pass the image through the model
        segm_map = model(image.cuda(), is_saved=False)

    # Plot the segmentation map
    plt.figure(figsize=(8, 8))
    plt.imshow(segm_map[0].cpu().numpy().transpose(1, 2, 0))  # Convert to HWC format
    plt.title('Segmentation Map')
    plt.axis('off')
    plt.show()


def main():
    # Example usage
    # Load your trained model
    model = GraphVariationalAutoencoder(mode="test", memory_bank_path="../model/results/model/memory_bank.index").cuda()
    # Load the model weights if you have them saved
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))

    # Test the model with an image
    image_path = (r'D:\UsingSpace\HCMUTE\Pratical Machine Learning and Artificial Intelligence'
                  '\patchcore-inspection-main\patchcore-inspection-main\mvtec\carpet\\test\color\\001.png')  # Replace with your image path
    test_and_plot_segmentation(model, image_path)


if __name__ == "__main__":
    main()
