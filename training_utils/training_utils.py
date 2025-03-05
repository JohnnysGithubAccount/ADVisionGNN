import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.functional import accuracy
from tqdm import tqdm
import gc

from GVAE.ADVisionGNN.utils.utils import instance_auroc, full_pixel_auroc


def train(
        train_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        loss_fn,
        optimizer
):
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    model = model.to(device)
    train_loss = 0
    for image in tqdm(train_loader, desc="Batches iterations"):
        image = image.to(device)

        output = model(image)

        loss = loss_fn(output, image)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()

    return train_loss / len(train_loader)

def validation(
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        loss_fn,
):
    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    model = model.to(device)
    test_loss = 0

    recon_error = []

    with torch.inference_mode():
        for image, mask, label in tqdm(test_loader, desc="[Test] Batches iterations"):
            image = image.to(device)

            output = model(image)

            loss = loss_fn(output, image)
            test_loss += loss

            torch.cuda.empty_cache()
            gc.collect()

            predicted_mask = ((image - output) ** 2).mean(axis=1)[:, 0: -10, 0:-10]

            recon_error.append(predicted_mask.mean(axis=(1, 2)))

    recon_error = torch.cat(recon_error).detach().cpu().numpy()

    best_threshold = np.mean(recon_error) + 3 * np.std(recon_error)

    total_sample = len(test_loader)

    return {
        "Test loss": test_loss / total_sample,
        "Best threshold": best_threshold
        # # "Instance auroc": auroc / total_sample,
        # "Pixel-wise auroc": pixel_wise_auroc / total_sample
    }


def main():
    pass


if __name__ == "__main__":
    main()
