from pathlib import Path
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from GVAE.ADVisionGNN.data.loader import create_loader
from torch import optim


def save_to_faiss(memory_list):
    pass


def subsampling(memory_list):
    selected_indicies = np.random.choice(
        len(memory_list),
        size=len(memory_list) // 10, replace=False
    )
    return memory_list[selected_indicies]


def main():
    pass


if __name__ == "__main__":
    main()
