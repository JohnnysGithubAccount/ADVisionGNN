import os
from sys import exec_prefix

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.optim import Adam, AdamW, Adamax, SGD, RMSprop
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from tqdm import tqdm

from GVAE.utils.utils import visualize_encoder_features
from utils.utils import ignore_warnings
from utils.utils import get_performance_plots
from utils.utils import set_up_pytorch
from utils.utils import get_image

from utils.mlflow_utils import get_and_set_experiment
from utils.mlflow_utils import get_signature
from utils.mlflow_utils import get_log_inputs

from training_utils.training_utils import train
from training_utils.training_utils import validation
from training_utils.function import ReconstructionLoss
from training_utils.configs import params

from data.loader import create_loader, MVTecDataset
from data.constant import *

from model.model import GraphVariationalAutoencoder

from typing import Dict, Tuple, List

import mlflow

import gc


def main():
    ignore_warnings()

    print(f"[INFO] Setting experiment")
    experiment_name: str = "Graph Neural Network based Variational Autoencoder"
    # experiment_name: str = "Graph Neural Network based Variational Autoencoder - Colored coffee can"
    artifact_path: str = "Main_Artifacts"

    experiment_id = get_and_set_experiment(
        experiment_name=experiment_name,
        artifact_path=artifact_path,
        tags={
            "Model": "Vision Graph Variational Autoencoder",
            "Purpose": "Design a version of Variational Autoencoder for Industrial Image Anomaly Detection"
        }
    )
    print(f"Experiment ID: {experiment_id}")

    print(f"[INFO] Getting Dataloader")
    main_folder_path = r"D:\UsingSpace\HCMUTE"\
                       r"\Pratical Machine Learning and Artificial Intelligence"\
                       r"\patchcore-inspection-main"\
                       r"\patchcore-inspection-main\mvtec"

    class_folders = [d for d in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, d))]

    train_image_transforms = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=tuple(TRAIN_MEAN), std=tuple(TRAIN_STD))
    ])

    test_image_transforms = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=tuple(TEST_MEAN), std=tuple(TEST_STD))
    ])

    mask_transforms = Compose([
        Resize((256, 256)),
        CenterCrop((224, 224)),
        ToTensor(),
    ])

    # set up the random seed for regeneration of the project purpose
    print(f"[INFO] Setting up PyTorch")
    device = set_up_pytorch(seed=params["seed"])
    print(f"[INFO] Using device: {device}")

    with mlflow.start_run(experiment_id=experiment_id, log_system_metrics=True) as main_run:

        mlflow.log_params(params=params)

        average_metric_saved = {
            "Loss": 0,
            "Instance AUROC": 0,
            "Full pixel AUROC": 0,
            "Anomaly pixel AUROC": 0
        }

        for class_name in class_folders[:]:
            with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=class_name) as inner_run:
                print(f"Implement the model")
                gvae = GraphVariationalAutoencoder()

                model_architecture_path = "model/model_architecture.txt"
                with open(model_architecture_path, "w") as file:
                    file.write(str(gvae))
                mlflow.log_artifact(
                    local_path=model_architecture_path,
                    artifact_path=artifact_path + "/artifacts/model_summary"
                )

                # # log the input dataset
                # input_df = get_log_inputs(
                #     image_dir=os.path.join(main_folder_path, class_name, "train", "good"),
                #     dataset_name=f" MVTec AD - {class_name}"
                # )
                # mlflow.log_input(dataset=input_df)

                # Create dataset object for a specific class
                dataset_path = os.path.join(main_folder_path, class_name)
                print(dataset_path)
                dataset = MVTecDataset(
                    source_path=dataset_path,
                    is_train=True,
                    transform=train_image_transforms
                )  # Replace 'class_name' with actual folder name
                train_loader = create_loader(
                    dataset=dataset,
                    batch_size=params["batch_size"],
                    shuffle=True
                )

                test_dataset = MVTecDataset(
                    source_path=dataset_path,
                    is_train=False,
                    transform=test_image_transforms,
                    mask_transform=mask_transforms
                )
                test_loader = create_loader(
                    dataset=test_dataset,
                    batch_size=params["batch_size"],
                    shuffle=False
                )

                print(f"[INFO] Setting Loss and Optimizers")
                loss_function = ReconstructionLoss(
                    alpha=1.0,
                    beta=1.0,
                    lambda_value=0.2
                ).to(device=device)

                # create and tracking optimizer for discriminator
                optimizer= torch.optim.RMSprop(
                    gvae.parameters(),
                    lr=params["learning_rate"]
                )
                params["optimizer"] = optimizer.__class__.__name__

                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode='min',
                    factor=.5,
                    patience=5,
                    min_lr=1e-15,
                    verbose=True
                )

                history = {
                    "Train loss": [],
                    "Test loss": []
                }

                for epoch in tqdm(range(params["num_epochs"])):
                    torch.cuda.empty_cache()
                    gc.collect()

                    print(f"Epoch: {epoch}")
                    train_loss = train(
                        train_loader=train_loader,
                        model=gvae,
                        device=device,
                        loss_fn=loss_function,
                        optimizer=optimizer
                    )
                    print(f"Train loss: {train_loss}")

                    torch.cuda.empty_cache()
                    gc.collect()

                    test_loss = validation(
                        test_loader=test_loader,
                        model=gvae,
                        device=device,
                        loss_fn=loss_function
                    )
                    print(f"Test loss: {test_loss}")

                    scheduler.step(test_loss["Test loss"])
                    current_lr = optimizer.param_groups[0]['lr']
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                    history["Train loss"].append(train_loss.item())
                    history["Test loss"].append(test_loss["Test loss"].item())

                    # log the losses in each epoch
                    mlflow.log_metric("Train loss", train_loss, step=epoch)
                    mlflow.log_metric("Test loss", test_loss["Test loss"], step=epoch)
                    # mlflow.log_metric("Instance AUROC", test_loss["Instance auroc"], step=epoch)
                    mlflow.log_metric("Best threshold", test_loss["Best threshold"], step=epoch)

                    torch.cuda.empty_cache()
                    gc.collect()

                    figure = get_image(
                        data_loader=test_loader,
                        model=gvae,
                        device=device,
                        best_threshold=test_loss["Best threshold"]
                    )
                    mlflow.log_figure(
                        figure=figure,
                        artifact_file=f"{artifact_path}/visualized/epoch{epoch}.png"
                    )
                    print("[INFO] Logged figure")

                    feature_maps = visualize_encoder_features(
                        test_loader=test_loader,
                        model=gvae,
                        device=device
                    )
                    mlflow.log_figure(
                        figure=feature_maps,
                        artifact_file=f"{artifact_path}/feature_maps/epoch{epoch}.png"
                    )
                    print("[INFO] Logged feature_maps")

                    # create discriminator's model signature
                    inputs = torch.randn(1, 3, 224, 224)
                    model_signature = get_signature(
                        inputs=inputs,
                        device=device,
                        model=gvae
                    )

                    # log the discriminator
                    mlflow.pytorch.log_model(
                        pytorch_model=gvae,
                        artifact_path=artifact_path + "/model" + gvae.__class__.__name__,
                        signature=model_signature
                    )
            folder_name = f"model/results/{class_name}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            model_result_path = os.path.join(folder_name, "model.pth")
            torch.save(gvae.state_dict(), model_result_path)

        # logged optimizer
        mlflow.log_metric("optimizer", params["optimizer"])

        # add some informative tags
        mlflow.set_tags(
            {
                "type": "GVAE",
                "dataset": "MVTEC AD",
            }
        )

        # add a brief description
        mlflow.set_tag(
            "mlflow.note.content",
            "This is  an anomaly detection project"
        )

if __name__ == "__main__":
    main()
