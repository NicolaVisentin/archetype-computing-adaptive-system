import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_mnist_data(
    root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10, permuted: bool = False
):
    """Get the MNIST dataset and return the train, validation and test dataloaders.

    Args:
        root (os.PathLike): Path to the folder containing the MNIST dataset.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        valid_perc (int): Percentage of the train dataset to use for
            validation. Defaults to 10.
        permuted (bool): Whether to apply permutation to images.
    """
    # Function to permutate the order of the pixels
    permutation = torch.randperm(28 * 28) if permuted else None 
    def permute_image(image): 
        if permutation is not None:
            image = image.view(-1)[permutation].view(1, 28, 28) 
        return image 
    
    # Transformations to apply to the images:
    #   1. Convert images into torch tensor. Shape (C, W, H) C=channels, W=width, H=heigh
    #   2. Apply custom operation (permutation defined above)
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Lambda(permute_image) 
    ])


    # Download (if not already there) full MNIST dataset (70 000 images) in specified folder and build torch
    # dataset for training (60 000 images) and torch dataset for testing (10 000 images). Also applies tranformations defined above
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, transform=transform
    )

    # Split train dataset into actual train dataset and validation dataset according to valid_perc
    valid_size = int(len(train_dataset) * (valid_perc / 100.0))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    # Build torch dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs_test, shuffle=False)

    return train_loader, valid_loader, test_loader
