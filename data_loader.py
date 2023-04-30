import os
import cv2
from PIL import Image
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision

import config


def TRAIN_loader(
    option="cifar10",
    shuffle=True,
    augment=True,
    is_glow=False,
    normalize=False,
    batch_size=1,
):
    """Return train_loader for given dataset"""
    """ Option : 'cifar10'  """

    preprocess = []

    if normalize:
        preprocess = add_normalize(preprocess)

    if option == "cifar10":
        if is_glow:
            opt = config.GLOW_cifar10
        else:
            opt = config.VAE_cifar10

        if augment:
            augment = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            augment = []

        dataset = dset.CIFAR10(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transforms.Compose(
                augment
                + [
                    transforms.Resize((opt.imageSize)),
                    transforms.ToTensor(),
                ]
                + preprocess
            ),
        )
        train_loader_cifar = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(opt.workers),
        )
        return train_loader_cifar


def TEST_loader(
    train_dist="cifar10",
    target_dist="cifar10",
    batch_size=1,
    shuffle=True,
    is_glow=False,
    normalize=False,
    is_pixelcnn=False,
):
    """Return test_loader for given 'train_dist' and 'target_dist'"""

    """ train_dist  """

    """ target_dist (In-Distribution or Out-of-Distribution)
    
    """

    preprocess = []

    if is_pixelcnn:
        rescaling = lambda x: (x - 0.5) * 2.0
        preprocess = [rescaling]

    if train_dist == "cifar10":
        opt = config.GLOW_cifar10 if is_glow else config.VAE_cifar10

        if target_dist == "cifar10":
            return test_loader_cifar10(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "svhn":
            return test_loader_svhn(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "lsun":
            return test_loader_lsun(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "mnist":
            return test_loader_mnist(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "fmnist":
            return test_loader_fmnist(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "kmnist":
            return test_loader_kmnist(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "omniglot":
            return test_loader_omniglot(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "notmnist":
            return test_loader_notmnist(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "noise":
            return test_loader_noise(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == "constant":
            return test_loader_constant(opt, preprocess, batch_size, shuffle, normalize)

        else:
            raise NotImplementedError("Oops! Such match of ID & OOD doesn't exist!")


def rgb_to_gray(x):
    return transforms.Grayscale(1)(x)


def gray_to_rgb(x):
    return x.repeat(3, 1, 1)


def add_normalize(preprocess, nc):
    if nc == 1:
        return preprocess + [transforms.Normalize((0.48,), (0.2,))]
    elif nc == 3:
        return preprocess + [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]


def test_loader_cifar10(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    gray_or_not = []
    if opt.nc == 1:
        gray_or_not += [rgb_to_gray]

    dataset_cifar10 = dset.CIFAR10(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose(
            gray_or_not
            + [
                transforms.Resize((opt.imageSize)),
                transforms.ToTensor(),
            ]
            + preprocess
        ),
    )
    test_loader_cifar10 = data.DataLoader(
        dataset_cifar10,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_cifar10


def test_loader_svhn(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    gray = []
    if opt.nc == 1:
        gray += [rgb_to_gray]
    dataset_svhn = dset.SVHN(
        root=opt.dataroot,
        split="test",
        download=True,
        transform=transforms.Compose(
            gray
            + [
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ]
            + preprocess
        ),
    )
    test_loader_svhn = data.DataLoader(
        dataset_svhn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_svhn


def test_loader_lsun(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    gray = []
    if opt.nc == 1:
        gray += [rgb_to_gray]

    transform = transforms.Compose(
        gray
        + [
            transforms.Resize(
                opt.imageSize
            ),  # Then the size will be H x 32 or 32 x W (32 is smaller)
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
        ]
        + preprocess
    )

    lsun = dset.ImageFolder(root="./data/LSUN_resize", transform=transform)
    test_loader_lsun = torch.utils.data.DataLoader(
        lsun, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return test_loader_lsun


def test_loader_mnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    rgb = []
    if opt.nc == 3:
        rgb += [gray_to_rgb]
    dataset_mnist = dset.MNIST(
        root=opt.dataroot + "/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ]
            + rgb
            + preprocess
        ),
    )
    test_loader_mnist = data.DataLoader(
        dataset_mnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_mnist


def test_loader_fmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    rgb = []
    if opt.nc == 3:
        rgb += [gray_to_rgb]
    dataset_fmnist = dset.FashionMNIST(
        root=opt.dataroot + "/fashionmnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((opt.imageSize)),
                transforms.ToTensor(),
            ]
            + rgb
            + preprocess
        ),
    )
    test_loader_fmnist = data.DataLoader(
        dataset_fmnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_fmnist


def test_loader_kmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    rgb = []
    if opt.nc == 3:
        rgb += [gray_to_rgb]
    dataset_kmnist = dset.KMNIST(
        root=opt.dataroot + "/kmnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ]
            + rgb
            + preprocess
        ),
    )
    test_loader_kmnist = data.DataLoader(
        dataset_kmnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_kmnist


def test_loader_omniglot(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    rgb = []
    if opt.nc == 3:
        rgb += [gray_to_rgb]
    dataset_omniglot = dset.Omniglot(
        root=opt.dataroot,
        background=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ]
            + rgb
            + preprocess
        ),
    )
    test_loader_omniglot = data.DataLoader(
        dataset_omniglot,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_omniglot


def test_loader_notmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    gray = []
    if opt.nc == 1:
        gray += [rgb_to_gray]

    class notMNIST(data.Dataset):
        def __init__(self, db_path, transform=None):
            super(notMNIST, self).__init__()
            self.db_path = db_path
            self.total_path = []
            alphabets = os.listdir(self.db_path)
            for alphabet in alphabets:
                path = self.db_path + "/" + alphabet
                elements = os.listdir(path)
                self.total_path += [path + "/" + element for element in elements]
            self.transform = transform

        def __len__(self):
            return len(self.total_path)

        def __getitem__(self, index):
            current_path = self.total_path[index]
            img = cv2.imread(current_path)
            img = Image.fromarray(img)
            img = self.transform(img)
            return img

    transform = transforms.Compose(
        gray
        + [
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ]
        + preprocess
    )

    notmnist = notMNIST(f"{opt.dataroot}/notMNIST_small/", transform=transform)
    test_loader_notmnist = data.DataLoader(
        notmnist,
        batch_size=batch_size,
        shuffle=shuffle,  # shuffle
        num_workers=0,
    )
    return test_loader_notmnist


def test_loader_noise(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)

    class Noise(data.Dataset):
        def __init__(self, number=10000, transform=None):
            super(Noise, self).__init__()
            self.transform = transform
            self.number = number
            self.total_data = np.random.randint(
                0, 256, (self.number, opt.nc, opt.imageSize, opt.imageSize)
            )

        def __len__(self):
            return self.number

        def __getitem__(self, index):
            array = torch.tensor(self.total_data[index] / 255).float()
            return self.transform(array)

    transform = transforms.Compose(preprocess)

    noise = Noise(transform=transform)
    test_loader_noise = data.DataLoader(
        noise,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_noise


def test_loader_constant(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)

    class Constant(data.Dataset):
        def __init__(self, number=10000, transform=None):
            super(Constant, self).__init__()
            self.number = number
            self.total_data = np.random.randint(0, 256, (self.number, opt.nc, 1, 1))
            self.transform = transform

        def __len__(self):
            return self.number

        def __getitem__(self, index):
            array = torch.tensor(self.total_data[index] / 255).float()
            array = array.repeat(1, opt.imageSize, opt.imageSize)
            return self.transform(array)

    transform = transforms.Compose(preprocess)

    constant = Constant(transform=transform)
    test_loader_constant = data.DataLoader(
        constant,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_constant
