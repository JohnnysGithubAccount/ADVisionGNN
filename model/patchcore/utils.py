from pathlib import Path
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import pairwise_distances
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

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the index to a file
    faiss.write_index(index, path)
    print(f"Features saved in FAISS database at '{path}'.")


def subsampling(memory_list, percent: int = 50):
    """

    :param memory_list:
    :param percent: 10 means 10%
    :return:
    """
    print(memory_list.shape)
    selected_indicies = np.random.choice(
        len(memory_list),
        size=len(memory_list) // percent,
        replace=False
    )
    print(memory_list[selected_indicies].shape)
    return memory_list[selected_indicies]


def corset_subsampling(memory_list, percent: int = 50):
    """
    Perform corset subsampling on the input data.
    :param memory_list: numpy array representing the data.
    :param percent: Percentage of data to keep (10 means 10%).
    :return: Subsampled memory_list using corset subsampling.
    """
    print(memory_list.shape)
    memory_list = memory_list.detach().cpu().numpy()

    # Determine the number of samples to retain
    num_samples_to_retain = len(memory_list) // percent

    # Calculate pairwise distances between all points
    distances = pairwise_distances(memory_list)

    # Initialize selected indices
    selected_indices = []
    remaining_indices = list(range(len(memory_list)))

    # Select the first point randomly
    selected_indices.append(remaining_indices.pop(np.random.choice(len(remaining_indices))))

    # Iteratively select points based on maximum distance to the current core-set
    for _ in range(num_samples_to_retain - 1):
        max_distances = np.min(distances[selected_indices][:, remaining_indices], axis=0)
        next_index = remaining_indices[np.argmax(max_distances)]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    print(memory_list[selected_indices].shape)
    return memory_list[selected_indices]


def main():
    pass


if __name__ == "__main__":
    main()
