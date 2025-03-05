import os

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.data import DatasetSource
from mlflow.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import torch


def get_and_set_experiment(
        experiment_name: str,
        artifact_path: str,
        tags: Dict[str, str]) -> str:
    """
    set the experiment for tracking using mlflow
    :param experiment_name:name of the experiment
    :param artifact_path: the name of the path that will save the artifacts
    :param tags: informative key: value format
    :return: experiment id
    """
    try:
        # try to create an experiment with the given name
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_path,
            tags=tags
        )
    except:
        # if the experiment already exist, get the experiment id
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name
        ).experiment_id
    finally:
        # finally, set the experiment using the experiment name
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_signature(
        inputs: torch.Tensor,
        device: str | torch.device,
        model: torch.nn.Module) -> mlflow.models.ModelSignature:

    model.eval()
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(inputs)  # predict from a sample input to get the output format

    model_signature = infer_signature(
        inputs.cpu().detach().numpy(),
        outputs.cpu().detach().numpy()
    )  # get the model signature

    return model_signature


def get_log_inputs(
        image_dir: str,
        dataset_name: str) -> Dataset:
    """
    Create a Dataset object in MLflow for tracking an image dataset.

    :param image_dir: Path to the directory containing image files.
    :param dataset_name: The name of the dataset to log.
    :return: MLflow Dataset object.
    """
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure there are images to log
    if not image_files:
        raise ValueError("No image files found in the specified directory.")

    # Create a DataFrame to log image paths if needed
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    # Create and return an MLflow Dataset object
    return Dataset(
        source=DatasetSource(),
        name=dataset_name
    )


def main():
    pass


if __name__ == "__main__":
    main()
