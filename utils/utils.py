import os
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import cv2
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score


def ignore_warnings() -> None:
    """
    ignore warnings for better visualization of the training process
    :return:
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    # this just for enabling logging system metrics if wanted
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def set_up_pytorch(seed: int = 42) -> torch.device:
    """
    set up random seed and get the current device
    :param seed: the seed you wanted to set
    :return: cuda or cpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_image(data_loader, model, device, best_threshold, figure_size: Tuple[int, int] = (10, 10)):
    image, mask, _ = next(iter(data_loader))
    print(image.shape)
    print(mask.shape)

    model.eval()
    with torch.inference_mode():
        image = image.to(device)
        model = model.to(device)

        reconstructed_image = model(image)

        # Calculate the predicted mask as the mean squared error
        # ((test_image - recon_image) ** 2).mean(axis=(1))[:, 0:-10, 0:-10].mean()
        predicted_mask = ((image - reconstructed_image) ** 2).mean(axis=1) * 10

    result = plt.figure(figsize=figure_size)

    # Original Image
    plt.subplot(3, 2, 1)
    plt.imshow(((image[0] + 1) / 2).permute(1, 2, 0).detach().cpu().numpy())
    plt.axis('off')
    plt.title('Input Image')

    # Reconstructed Image
    plt.subplot(3, 2, 2)
    plt.imshow((((reconstructed_image[0] + 1) / 2).permute(1, 2, 0).detach().cpu().numpy()))
    plt.axis('off')
    plt.title('Reconstructed Image')

    # Anomaly score
    plt.subplot(3, 2, 3)
    plt.imshow(predicted_mask[0].detach().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title("Anomaly Score")

    # True Mask
    plt.subplot(3, 2, 4)
    plt.imshow(mask[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title("Ground Truth")

    # Heatmap for Predicted Mask
    plt.subplot(3, 2, 5)
    # Normalize predicted mask for heatmap
    predicted_mask_normalized = (predicted_mask[0]).detach().cpu().numpy()
    plt.imshow(predicted_mask_normalized,
               cmap='jet',
               vmin=predicted_mask_normalized.min(),
               vmax=predicted_mask_normalized.max())
    plt.colorbar()
    plt.axis("off")
    plt.title("Predicted (Heatmap)")

    # Otsu's Thresholding for Predicted Mask
    predicted_mask_np = (predicted_mask[0].detach().cpu().numpy()).astype(np.uint8)
    thresholded = 1 * (predicted_mask_np >= best_threshold)


    plt.subplot(3, 2, 6)
    thresholded_mask = (predicted_mask_np > thresholded).astype(float)  # Convert to binary mask
    plt.imshow(thresholded_mask, cmap='gray')
    plt.axis("off")
    plt.title("Predicted (Threshold)")

    plt.tight_layout()
    return result

def get_performance_plots(
        history: Dict[str, list]) -> plt.Figure:
    """
    get the loss and accuracy curves of the model to track using mlflow
    :param history: dictionary of training history
    :return: dictionary of curves
    """
    loss_curve = plt.figure()
    plt.plot(
        np.arange(len(history["Train loss"])),
        history["Train loss"],
        label="Training Loss"
    )
    plt.plot(
        np.arange(len(history["Test loss"])),
        history["Test loss"],
        label="Validation Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    return loss_curve


def visualize_encoder_features(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            images, _, _ = data  # Assuming the DataLoader returns (images, labels)
            images = images.to(device)

            # Pass the images through the encoder
            stem_output = model.stem(images)
            encoder_output = model.encoder(stem_output)

            # Get the feature maps from the encoder output
            break  # For demonstration, we'll visualize the first batch

    # Plotting the feature maps
    num_features = encoder_output.size(1)  # Number of feature maps
    feature_map_size = encoder_output.size(2)  # Height/Width of each feature map

    # Determine how many feature maps to plot (up to 25)
    num_to_plot = min(num_features, 25)

    # Setting up the grid for subplots (5 rows, 5 columns)
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i in range(num_to_plot):
        # Get the feature map, squeeze out the batch and channel dimensions
        feature_map = encoder_output[0, i].cpu().numpy()  # Get the first sample

        # Rescale feature map to range [0, 1]
        feature_map_min = feature_map.min()
        feature_map_max = feature_map.max()
        if feature_map_max > feature_map_min:  # Avoid division by zero
            feature_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min)

        axes[i].imshow(feature_map, cmap='viridis')  # Use a colormap
        axes[i].axis('off')  # Turn off axis labels
        axes[i].set_title(f'Feature Map {i + 1}')  # Title for each subplot

    # Hide any unused subplots
    for j in range(num_to_plot, 25):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def full_pixel_auroc(output: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Calculate the pixel-wise AUROC for a batch of images.

    Args:
        output (torch.Tensor): The model's output (predicted probabilities).
        mask (torch.Tensor): The ground truth binary mask.

    Returns:
        float: The calculated AUROC for the current instance.
    """
    # Flatten the tensors
    output_flat = output.view(-1).cpu().detach().numpy()
    mask_flat = mask.view(-1).cpu().detach().numpy()

    # Ensure the mask is binary
    if mask_flat.max() > 1 or mask_flat.min() < 0:
        raise ValueError("Mask should be binary (0s and 1s).")

    # Calculate AUROC
    return roc_auc_score(mask_flat, output_flat)



def anomaly_pixel_auroc():
    pass


def instance_auroc(reconstructed: torch.Tensor, original: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate class AUROC for anomaly detection based on reconstruction error.

    Args:
        reconstructed (torch.Tensor): The reconstructed images from the model (shape: (16, 3, 224, 224)).
        original (torch.Tensor): The original images (shape: (16, 3, 224, 224)).
        labels (torch.Tensor): Ground truth binary labels (shape: (16,)).

    Returns:
        float: The calculated instance AUROC for the batch.
    """
    # Calculate the reconstruction error (Mean Squared Error)
    mse = torch.mean((reconstructed - original) ** 2, dim=(1, 2, 3))  # Shape: (16,)
    print(mse.shape)

    # Flatten labels to match the batch size
    labels_flat = labels.view(-1).cpu().detach().numpy()  # Shape: (16,)
    print(len(labels))

    # Calculate AUROC
    return roc_auc_score(labels_flat, mse.cpu().detach().numpy())


def calculate_accuracy():
    pass


def main():
    pass



if __name__ == "__main__":
    main()
