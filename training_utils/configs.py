from GVAE.data.constant import *
from typing import Dict
import torch


params: Dict[str, float | int | str] = {
    "batch_size": 8,
    "train_mean": f"{TRAIN_MEAN}",
    "train_std": f"{TRAIN_STD}",
    "test_mean": f"{TEST_MEAN}",
    "test_std": f"{TEST_STD}",
    "seed": 111,
    "device": torch.cuda.get_device_name(torch.cuda.current_device()),
    "learning_rate": 0.0002,
    "num_epochs": 40,
    "loss_function": "MSE + Structural Similarity Index Loss(SSIM)",

}