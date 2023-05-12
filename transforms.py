import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def get_transforms_testing(opt, grayscale=False):
    tfms = []
    if grayscale:
        tfms.append(transforms.Grayscale(1))
    tfms.append(transforms.Resize([opt.load_size, opt.load_size], Image.BICUBIC))
    tfms.append(transforms.Lambda(lambda img: make_power_2(img, base=4)))
    tfms += [transforms.ToTensor()]
    if grayscale:
        tfms += [transforms.Normalize((0.5,), (0.5,))]
    else:
        tfms += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(tfms)


def add_noise(image, mean=0, std=0.1):
    np_img = np.array(image)
    channels = np_img.shape[-1]
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def get_transforms_training(opt, grayscale=False):
    tfms = []
    if grayscale:
        tfms.append(transforms.Grayscale(1))
    tfms.append(transforms.Resize([opt.load_size, opt.load_size], Image.BICUBIC))
    tfms.append(transforms.Lambda(lambda img: make_power_2(img, base=4)))
    tfms.append(transforms.RandomHorizontalFlip(p=0.5))

    tfms += [transforms.ToTensor()]
    if grayscale:
        tfms += [transforms.Normalize((0.5,), (0.5,))]
    else:
        tfms += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(tfms)


def make_power_2(img, base):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), Image.BICUBIC)


# tfms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
# tfms.append(transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)))
