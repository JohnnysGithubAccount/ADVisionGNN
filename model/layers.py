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
from GVAE.ADVisionGNN.model.patchcore.utils import save_to_faiss, subsampling, corset_subsampling, load_faiss_to_tensor


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

#
# class CNNDecoder(nn.Module):
#     """Normal CNN Decoder"""
#     def __init__(self,
#                  hidden_channels: List,
#                  kernel_sizes: List,
#                  list_strides: List,
#                  paddings: List
#                  ):
#         super().__init__()
#         # [1, 384, 7, 7]
#         self.conv_layers = nn.ModuleList([])
#         for i in range(len(hidden_channels) - 1):
#             self.conv_layers.append(nn.Sequential(
#                 nn.ConvTranspose2d(
#                     in_channels= hidden_channels[i],
#                     out_channels=hidden_channels[i + 1],
#                     kernel_size=kernel_sizes[i],
#                     stride=list_strides[i],
#                     padding=paddings[i]
#                 ),
#                 nn.BatchNorm2d(hidden_channels[i + 1]),
#                 nn.ReLU()
#             ))
#
#         self.output_layer = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=hidden_channels[-1],
#                 out_channels=3,
#                 kernel_size=kernel_sizes[-1],
#                 padding=paddings[-1],
#                 stride=list_strides[-1]
#             ),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         for layer in self.conv_layers:
#             x = layer(x)
#         x = self.output_layer(x)
#         return x


import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CNNDecoder(nn.Module):
    """Normal CNN Decoder"""
    def __init__(self,
                 hidden_channels: List[int],
                 kernel_sizes: List[int],
                 list_strides: List[int],
                 paddings: List[int]
                 ):
        super().__init__()
        # [1, 384, 7, 7]
        self.conv_layers = nn.ModuleList([])
        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=list_strides[i],
                    padding=paddings[i]
                ),
                nn.BatchNorm2d(hidden_channels[i + 1]),
                nn.ReLU()
            ))

        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=list_strides[-1], mode='bilinear', align_corners=True),
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    decoder = CNNDecoder(
        hidden_channels=[384, 240, 96, 3],
        kernel_sizes=[4, 4, 4, 4],
        list_strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 1]
    ).cuda()

    decoder.eval()
    with torch.inference_mode():
        output = decoder(torch.rand((1, 384, 14, 14)).cuda())
        print(output.shape)