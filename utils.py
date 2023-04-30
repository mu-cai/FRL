import torch.nn.functional as F
import torch
from torch.autograd import Variable
import cv2


mse = torch.nn.MSELoss()


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)


def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel


def find_pixel_high_freq(im, gauss_kernel, find=False, index=None, is_gray=False):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    if is_gray:
        im_gray = im[:, 0, ...]
    else:
        im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    return (im_gray - low_gray)[:, 0:1, :, :]


def process_x(x, args):
    if args.train_dist == "fmnist":
        is_gray = True
    else:
        is_gray = False

    gauss_kernel = get_gaussian_kernel(args.gauss_size).cuda()
    return (find_pixel_high_freq(x, gauss_kernel, is_gray=is_gray) + 1) / 2


def process_target(x, args=None):
    return Variable(x.data.view(-1) * 255).long()


def produce_concat_x(x, opt):
    x_H_org = process_x(x, opt)
    x = torch.cat((x, x_H_org), dim=1)
    return x
