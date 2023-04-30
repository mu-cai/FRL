import argparse
import random
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import os, sys
from pathlib import Path
from torch import distributed as dist
import train_VAE.DCGAN_VAE_freq as DVAE

path = Path(os.getcwd()).parent
sys.path.append(str(path))
from utils import process_x, process_target
import torch.utils.data as data
from tqdm import tqdm


criterion_L1 = torch.nn.L1Loss()


def KL_div(mu, logvar, reduction="avg"):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="../data", help="path to dataset")
    parser.add_argument( "--workers", type=int, help="number of data loading workers", default=4)
    parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--imageSize",
        type=int,
        default=32,
        help="the height / width of the input image to network",
    )
    parser.add_argument("--nc", type=int, default=3, help="input image channels")
    parser.add_argument(
        "--nz", type=int, default=200, help="size of the latent z vector"
    ) 
    parser.add_argument("--ngf", type=int, default=64, help="hidden channel sieze")
    parser.add_argument(
        "--num_epoch", type=int, default=100, help="number of epochs to train for"
    )  
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="beta for beta-vae")

    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--gauss_size", type=int, default=5)
    parser.add_argument("--cuda_num", type=str, default="1", help="")
    parser.add_argument("--dataset", type=str, default="cifar10", help="")
    parser.add_argument("--print_text", action="store_true", help="")
    parser.add_argument("--model_save_num", type=int, default=20)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--seed_val", type=int, default=-1)

    opt = parser.parse_args()
    cuda_num = opt.cuda_num
    if len(opt.cuda_num) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.seed_val != -1:
        opt.manualSeed = opt.seed_val  # random.randint(1, 10000) # fix seed
    else:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    #############################################################
    #############################################################
    #############################################################

    opt.train_dist = opt.dataset

    # augment = "hflip"
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ]
    )

    if opt.dataset == "cifar10":
        opt.image_channel = 3
        opt.nc = opt.image_channel + 1
        opt.num_epoch = 100
        opt.ngf = 64
        experiment = "../saved_models/VAE_cifar10"
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=True,
            train=True,
            transform=transform,
        )

    elif opt.dataset == "fmnist":
        opt.image_channel = 1
        opt.nc = opt.image_channel + 1
        opt.num_epoch = 100
        opt.ngf = 32
        experiment = "../saved_models/VAE_fmnist"
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            download=True,
            train=True,
            transform=transform,
        )

    else:
        raise ValueError("Oops! Dataset is incorrect! Bye~")

    if len(opt.cuda_num) == 1:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
        )

    if not os.path.exists(experiment):
        os.mkdir(experiment)
    name = experiment.split("/")[-1]
    print(f"Dataloader for {name} is ready!")
    print(f'Please see the path "{experiment}" for the saved model !')

    #############################################################
    #############################################################
    #############################################################

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    print(f"Channel {nc}, ngf {ngf}, nz {nz}")
    beta = opt.beta

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    netG.apply(weights_init)

    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    netE.apply(weights_init)

    if len(opt.cuda_num) > 1:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        def synchronize():
            if not dist.is_available():
                return
            if not dist.is_initialized():
                return
            world_size = dist.get_world_size()
            if world_size == 1:
                return
            dist.barrier()

        synchronize()
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        netE = nn.parallel.DistributedDataParallel(
            netE,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        netG = nn.parallel.DistributedDataParallel(
            netG,
            device_ids=[0],
            output_device=0,
            broadcast_buffers=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            sampler=data.distributed.DistributedSampler(dataset, shuffle=True),
            drop_last=True,
        )

    # setup optimizer

    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay=0)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.5)

    netE.to(device)
    netG.to(device)

    start_epoch = 0
    if opt.resume:
        save_name_pth = (
            experiment
            + f"/net_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_epoch_{opt.resume_epoch}.pth"
        )
        checkpoint = torch.load(save_name_pth, map_location=device)
        state_G = checkpoint["state_dict_G"]
        state_E = checkpoint["state_dict_E"]
        optimizer1.load_state_dict(checkpoint["optimizer1"])
        optimizer2.load_state_dict(checkpoint["optimizer2"])
        scheduler1.load_state_dict(checkpoint["scheduler1"])
        scheduler2.load_state_dict(checkpoint["scheduler2"])
        start_epoch = checkpoint["epoch"] + 1
        netG.load_state_dict(state_G)
        netE.load_state_dict(state_E)

    print("start_epoch :", start_epoch)

    netE.train()
    netG.train()

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    rec_l = []
    kl = []
    history = []
    start = datetime.today()

    for epoch in range(start_epoch, opt.num_epoch):
        print(f"/net_pixel_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_epoch_{epoch+1}.pth")
        mean_loss = 0.0
        i = 0
        for x, _ in tqdm(dataloader):
            i += 1
            x = x.to(device)
            x_H_org = process_x(x, opt)

            x = torch.cat((x, x_H_org), dim=1)

            b = x.size(0)
            target = process_target(x)
            [z, mu, logvar] = netE(x)
            recon = netG(z)

            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b

            kld = KL_div(mu, logvar)
            loss = recl + opt.beta * kld.mean()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward(retain_graph=True)

            optimizer1.step()
            optimizer2.step()
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())
            mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)

            if not i % 100:
                print(f"epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f} ")
                if opt.print_text:
                    txt = open(
                        experiment + f"/ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_epoch.txt",
                        "a",
                    )
                    txt.writelines(
                        f"epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f}"
                    )
                    txt.writelines("\n")
                    txt.close()

        history.append(mean_loss)
        scheduler1.step()
        scheduler2.step()
        now = datetime.today()
        print(f"\nNOW : {now:%Y-%m-%d %H:%M:%S}, Elapsed Time : {now - start}\n")
        if epoch % opt.model_save_num == opt.model_save_num - 1:
            save_dict = {
                "epoch": epoch,
                "state_dict_E": netE.state_dict(),
                "state_dict_G": netG.state_dict(),
                "optimizer1": optimizer1.state_dict(),
                "optimizer2": optimizer2.state_dict(),
                "scheduler1": scheduler1.state_dict(),
                "scheduler2": scheduler2.state_dict(),
            }
            save_name_pth = (
                experiment
                + f"/net_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_epoch_{epoch+1}.pth"
            )
            torch.save(save_dict, save_name_pth)


if __name__ == "__main__":
    main()
