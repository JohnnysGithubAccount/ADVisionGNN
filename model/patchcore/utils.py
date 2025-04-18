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
import faiss


def save_to_faiss(features, path: str):    # Ensure features are in float32 format
    features = features.astype('float32')

    # Create a FAISS index using L2 distance
    n_features = features.shape[1]
    index = faiss.IndexFlatL2(n_features)

    # Add features to the index
    index.add(features)

    # Save the index to a file
    faiss.write_index(index, path)
    print(f"Features saved in FAISS database at '{path}'.")


def subsampling(memory_list, percent: int = 10):
    """

    :param memory_list:
    :param percent: 10 means 10%
    :return:
    """
    selected_indicies = np.random.choice(
        len(memory_list),
        size=len(memory_list) // percent, replace=False
    )
    return memory_list[selected_indicies]


def main():
    pass


if __name__ == "__main__":
    main()
