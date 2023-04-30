class VAE_cifar10:
    train_dist = "cifar10"
    ood_list = [
        "cifar10",
        "svhn",
        "lsun",
        "mnist",
        "fmnist",
        "kmnist",
        "omniglot",
        "notmnist",
        "noise",
        "constant",
    ]
    dataroot = "./data"
    modelroot = "./saved_models"
    workers = 0
    imageSize = 32
    nc = 3  # Num of c (channels)
    nz = 200  # Num of z (latent)
    ngf = 64  # Num of Generator Filter size (scaling factor)
    ngpu = 1
    beta = 1
    batch_size = 1
    train_batchsize = 1
    num_samples = 1
