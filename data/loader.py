import pathlib
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


class MVTecDataset(Dataset):
    def __init__(self,
                 source_path: str,
                 transform = None,
                 mask_transform = None,
                 is_train: bool = True,
                 image_size: int = 224) -> None:
        self.source = source_path
        self.is_train = is_train
        self.image_size = image_size

        if is_train:
            self.images_dir = (list(pathlib.Path(os.path.join(source_path, 'train', 'good')).glob("*.png"))
                               + list(pathlib.Path(os.path.join(source_path, 'train', 'good')).glob("*.jpg")))
        else:
            self.images_dir = (list(pathlib.Path(os.path.join(source_path, 'test')).glob("*/*.png"))
                               + list(pathlib.Path(os.path.join(source_path, 'test')).glob("*/*.jpg")))
            self.mask_dir = (list(pathlib.Path(os.path.join(source_path, 'ground_truth')).glob("*/*.png"))
                             + list(pathlib.Path(os.path.join(source_path, 'ground_truth')).glob("*/*.jpg")))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.images_dir)

    def load_image(self, index: int) -> Image.Image:
        img_path = self.images_dir[index]
        if 'good' in str(img_path) and not self.is_train :
            self.mask_dir.insert(index, 'good_mask')
        return Image.open(img_path).convert('RGB')

    def load_mask(self, index: int) -> Image.Image:
        if 'good' in str(self.images_dir[index]):
            return Image.new('L', (self.image_size,) * 2, color=0)

        if self.mask_dir[index] == 'good_mask':
            return Image.new('L', (self.image_size,) * 2, color=0)

        mask_path = self.mask_dir[index]
        return Image.open(mask_path)


    def get_label(self, index):
        if 'good' in str(self.images_dir[index]):
            return 0
        return 1


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = self.load_image(index)
        if self.transform:
            try:
                img = self.transform(img)
            except:
                print(self.images_dir[index])

        if not self.is_train:
            mask = self.load_mask(index)

            if self.mask_transform:
                mask = self.mask_transform(mask)

            label = self.get_label(index)

            return img, mask, label
        else:
            return img


def create_loader(dataset, batch_size: int = 32, shuffle: bool = False) -> torch.utils.data.DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader


def calculate_normalization_params(data_loader, is_train: bool = False):
    mean = 0.0
    std = 0.0
    total_images = 0
    for batch in tqdm(data_loader):
        if is_train:
            images = batch
        else:
            images = batch[0]
        batch_samples = images.size(0)  # batch size (number of images in the batch)
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height * width)

        mean += images.mean(dim=2).sum(dim=0)  # sum up the means for each channel
        std += images.std(dim=2).sum(dim=0)  # sum up the stds for each channel
        total_images += batch_samples

    # Final mean and std
    mean /= total_images
    std /= total_images
    return mean, std


def main():
    check_train = True
    check_test = True
    cal_mean_std = True

    main_folder_path = (r"D:\UsingSpace\HCMUTE"
                        r"\Pratical Machine Learning and Artificial Intelligence"
                        r"\patchcore-inspection-main"
                        r"\patchcore-inspection-main\mvtec")
    class_folders = [d for d in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, d))]
    print(class_folders)

    # Augment train data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataset object for a specific class, e.g., 'bottle'
    dataset = MVTecDataset(
        source_path=os.path.join(main_folder_path, class_folders[1]),
        is_train=True,
        transform=train_transforms
    )
    train_loader = create_loader(
        dataset=dataset,
        batch_size=16,
        shuffle=True
    )

    test_dataset = MVTecDataset(
        source_path=os.path.join(main_folder_path, class_folders[1]),
        is_train=False,
        transform=train_transforms,
        mask_transform=train_transforms
    )
    test_loader = create_loader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False
    )

    if cal_mean_std:
        print(f"Mean, std on train: {calculate_normalization_params(train_loader, is_train=True)}")
        print(f"Mean, std on test: {calculate_normalization_params(test_loader, is_train=False)}")


    # Check dataset object
    if check_train:
        print(f"Number of images in dataset: {len(dataset)}")
        for images in train_loader:
            print(images.shape)  # Print the shape of the batch of images
            plt.imshow(images[0].permute(1, 2, 0))
            plt.title('Sample training image')
            plt.show()
            break  # Remove this if you want to iterate through the entire DataLoader
    if check_test:
        for images, masks in test_loader:
            print(images.shape)
            print(masks.shape)
            plt.subplot(1, 2, 1)
            plt.imshow(images[0].permute(1, 2, 0))
            plt.title('Test image')
            plt.subplot(1, 2, 2)
            plt.imshow(masks[0].permute(1, 2, 0))
            plt.title('Mask')
            plt.show()
            break

if __name__ == "__main__":
    main()
