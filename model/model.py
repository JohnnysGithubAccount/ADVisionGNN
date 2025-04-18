import math

import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from typing import List, Tuple, Dict
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from GVAE.ADVisionGNN.gcn_lib import Grapher, act_layer
from patchcore.utils import save_to_faiss, subsampling


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
                 memory_bank_path: str = "../model/results/model/memory_bank.index"):
        """

        :param decoder: cnn or graph
        :param mode: train or test mode
        """
        super(GraphVariationalAutoencoder, self).__init__()

        self.memory_bank = list()

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

        self.model.encoder.block[4][1].fc2[1].register_forward_hook(hook)
        self.model.encoder.block[7][1].fc2[1].register_forward_hook(hook)

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

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape

        x = self.encoder(x)
        x = self.decoder(x)

        

        return x


class PatchExtractor(torch.nn.Module):
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def forward(self, feature_maps):
        # Validate the shape of the input feature maps
        if feature_maps.shape != (1, 240, 14, 14):
            raise ValueError("Input feature maps must have shape (1, 240, 14, 14)")

        # Create patches of shape (240, 3, 3)
        patches = []
        for i in range(0, 14 - 3 + 1):  # Iterate over rows
            for j in range(0, 14 - 3 + 1):  # Iterate over columns
                patch = feature_maps[:, :, i:i + 3, j:j + 3]  # Keep batch dimension
                patches.append(patch)

        # Convert patches to a tensor
        patches = torch.stack(patches)  # Shape will be (number of patches, 1, 240, 3, 3)

        # Reshape for FAISS, we need to flatten the patches
        num_patches = patches.shape[0]
        patches_flat = patches.view(num_patches, 240 * 3 * 3)  # Shape will be (num_patches, 240 * 3 * 3)

        # Initialize a FAISS index
        d = 240 * 3 * 3  # Dimension of each patch
        index = faiss.IndexFlatL2(d)  # Using L2 distance

        # Add patches to the FAISS index
        index.add(patches_flat.detach().numpy().astype(float))

        # Save the FAISS index to disk
        faiss.write_index(index, 'feature_patches.index')

        print("Patches saved to FAISS index successfully.")


def main():
    n_blocks = sum([2, 2, 6, 2])
    channels = [48, 96, 240, 384]
    act = 'relu'

    stem = Stem(out_dim=channels[0], act=act)

    encoder = GraphEncoder(
        num_block=[2, 2, 2],
        hidden_channels=[48, 96, 240],
        reduce_ratios=[4, 2, 1],
        dpr = [x.item() for x in torch.linspace(0, 0.0, n_blocks)]
    )

    # Create dummy feature maps
    # feature_maps = torch.rand(240, 14, 14).float()
    extractor = PatchExtractor()

    device = torch.device('cpu')
    print(f"Using: {device}")

    inputs = torch.randn(1, 3, 224, 224).to(device)
    print(inputs.shape)

    stem = stem.to(device)
    encoder = encoder.to(device)
    extractor.to(device)

    # forward
    stem_output = stem(inputs)
    encoder_output = encoder(stem_output)
    print(encoder_output.shape)
    # extractor(encoder_output)
    index = faiss.read_index('feature_patches.index')
    # Check the type of index
    print("Index type:", type(index))

    # Check the dimension of the index
    print("Dimension of the index:", index.d)

    # Number of vectors stored in the index
    print("Number of vectors in the index:", index.ntotal)

    # Example of performing a search
    # For demonstration, create a random query vector
    query_vector = np.random.rand(1, index.d).astype(np.float32)
    distances, indices = index.search(query_vector, k=5)  # Find the 5 nearest neighbors
    print("Distances to nearest neighbors:", distances)
    print("Indices of nearest neighbors:", indices)



if __name__ == "__main__":
    main()
