import argparse, random
import cv2
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import train_VAE.DCGAN_VAE_freq as DVAE
from data_loader import TEST_loader
import config
from datetime import datetime
import torch.nn.functional as F
from utils import produce_concat_x, process_target
from sklearn import metrics


def process_all_score(score):
    for i in range(len(score)):
        score[i] = process_only_nan(score[i])
    return score


def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(
            torch.log(torch.mean(torch.exp(weights - weights.max()))) + weights.max()
        )
    return NLL_loss


def store_NLL(x, recon, mu, logvar, z, opt=None):
    with torch.no_grad():
        sigma = torch.exp(0.5 * logvar)
        b = x.size(0)
        log_p_z = -torch.sum(z**2 / 2 + np.log(2 * np.pi) / 2, 1)
        z_eps = (z - mu) / sigma
        z_eps = z_eps.view(opt.repeat, -1)
        log_q_z_x = -torch.sum(z_eps**2 / 2 + np.log(2 * np.pi) / 2 + logvar / 2, 1)

        target = process_target(x)
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        cross_entropy = F.cross_entropy(recon, target, reduction="none")
        log_p_x_z_org = -torch.sum(cross_entropy.view(b, -1), 1)
        weights_org = log_p_x_z_org + log_p_z - log_q_z_x
    return weights_org


def process_only_nan(NLL_loss):
    if np.isnan(NLL_loss):
        NLL_loss = 1e30
    return NLL_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", help="")
    parser.add_argument("--trade_off_ratio", type=float, default=1)
    parser.add_argument("--gauss_size", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=5000)
    parser.add_argument("--seed_val", type=int, default=2021)

    args = parser.parse_args()
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "cifar10":
        opt = config.VAE_cifar10
        opt.repeat = 20
        opt.state_dict = (
            "./saved_models/VAE_cifar10/net_ngf_64_nz_200_beta_1.0_epoch_100.pth"
        )

    opt.train_dist = args.dataset
    nc = int(opt.nc) + 1  # frequency channel
    opt.gauss_size = args.gauss_size

    auroc_list = []

    ngpu = int(opt.ngpu)
    nz = 200
    ngf = int(opt.ngf)

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    checkpoint = torch.load(opt.state_dict, map_location=device)
    state_G = checkpoint["state_dict_G"]
    state_E = checkpoint["state_dict_E"]
    netG.load_state_dict(state_G)
    netE.load_state_dict(state_E)
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    ######################################################################

    for ood_index, ood in enumerate(opt.ood_list):
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=False,
        )

        difference = []
        if ood == opt.train_dist:
            start = datetime.now()

        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            x = x.expand(opt.repeat, -1, -1, -1).contiguous()
            weights_agg = []
            with torch.no_grad():
                x = x.to(device)
                x = produce_concat_x(x, opt)
                b = x.size(0)
                [z, mu, logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z, opt=opt)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1)
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

                img = x[0, : opt.nc, :, :].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img_encoded = cv2.imencode(
                    ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                )
                L = len(img_encoded[1]) * 8
                difference.append(NLL_loss - args.trade_off_ratio * L)
            if len(difference) == args.test_num:
                break

        if ood == opt.train_dist:
            end = datetime.now()
            avg_time = (end - start).total_seconds() / args.test_num

        difference = process_all_score(difference)
        if opt.train_dist == ood:
            indist = difference
            label1 = np.ones(len(indist))
        ood_ = difference
        combined = np.concatenate((indist, ood_))
        label2 = np.zeros(len(ood_))
        label = np.concatenate((label1, label2))
        fpr, tpr, _ = metrics.roc_curve(label, combined, pos_label=0)
        auroc = metrics.auc(fpr, tpr)
        auroc_list.append(auroc)
        if ood_index > 0:
            print(f"{ood}: {auroc}")

    print("AVG AUROC:", np.average(np.asarray(auroc_list[1:])))
    print(f"Average {opt.train_dist} inference time : {avg_time} seconds")
    print(f"Average #images processed : {1 / avg_time} images")
