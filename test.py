import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model.model import GraphVariationalAutoencoder
from PIL import Image


# Load the model
model_path = r"model/results/bottle/model.pth"
model = GraphVariationalAutoencoder()

# Load the model state dictionary
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

# Define the path to your image
image_path = r"D:\UsingSpace\HCMUTE\Pratical Machine Learning and Artificial Intelligence\patchcore-inspection-main\patchcore-inspection-main\mvtec\bottle\train\good\000.png"

# Load the image
image = Image.open(image_path)

# Define transformations (resize, center crop, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),  # Resize to match model input
    transforms.ToTensor(),                # Convert to tensor
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize based on your model's requirements
])

# Apply transformations
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Pass the image through the model
with torch.inference_mode():
    output = model(image_tensor)

# Convert the output tensor back to an image
output_image_tensor = output.squeeze(0)  # Remove batch dimension
output_image_tensor = (output_image_tensor * 0.5) + 0.5  # Denormalize to [0, 1]
output_image_np = transforms.ToPILImage()(output_image_tensor.cpu())

# Prepare to plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the input image
axes[0].imshow(image)  # Original image
axes[0].axis('off')  # Turn off axis numbers and ticks
axes[0].set_title('Input Image')

# Plot the output image
axes[1].imshow(output_image_np)  # Output image
axes[1].axis('off')  # Turn off axis numbers and ticks
axes[1].set_title('Output Image')

# Show the plot
plt.tight_layout()
plt.show()