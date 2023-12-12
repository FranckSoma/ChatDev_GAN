'''
This file handles data loading and preprocessing for the GAN.
'''
import torch
from torchvision import datasets, transforms
def get_data_loader(batch_size, dataset='MNIST'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if dataset == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    elif dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    else:
        raise ValueError("Invalid dataset name. Supported datasets: 'MNIST', 'CIFAR10'")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader