import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure

from skimage.metrics import structural_similarity as ssim
import numpy as np


class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0, block_size: int = 8):
        super().__init__()

        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.block_size = block_size

    def forward(self, inputs, targets):
        batch_size, channels, height, width = inputs.shape

        num_blocks_y = height // self.block_size
        num_blocks_x = width // self.block_size

        ssim_values = []

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                start_y = i * self.block_size
                start_x = j * self.block_size
                end_y = start_y + self.block_size
                end_x = start_x + self.block_size

                y_true_block = inputs[:, :, start_y:end_y, start_x:end_x]
                y_predicted_block = targets[:, :, start_y:end_y, start_x:end_x]

                ssim_value = self.ssim(y_predicted_block, y_true_block)
                ssim_values.append(1 - ssim_value)

        return torch.stack(ssim_values).mean()



class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, lambda_value: float = 1.0):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE + SSIM
        self.beta = beta  # Weight for entropy loss
        self.lambda_value = lambda_value

        self.mse_loss = nn.MSELoss()

        self.ssim = SSIMLoss()

        self.entropy_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        # Compute MSE loss
        mse_loss = self.mse_loss(inputs, targets)

        # Compute SSIM loss
        ssim_loss = self.ssim(inputs, targets)

        # Combine MSE and SSIM loss
        reconstruction_loss = mse_loss + self.lambda_value * ssim_loss


        # Final combined loss
        total_loss = self.alpha * reconstruction_loss

        return total_loss


def main():
    # Assuming 'recon_x' is the reconstructed output and 'x' is the target input
    loss_fn = ReconstructionLoss(alpha=1.0, beta=0.5)
    recon_x = torch.randn(1, 3, 224, 224)  # Example reconstructed image
    x = torch.randn(1, 3, 224, 224)  # Example target image

    loss = loss_fn(x, recon_x)
    print("Total Loss:", loss.item())


if __name__ == "__main__":
    main()
