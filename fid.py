"""
Copyright (c) 2023, Andreas Øie
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import warnings
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from halo import Halo
from scipy import linalg
from torchvision import models
from tqdm import tqdm

warnings.filterwarnings("ignore")

from PIL import Image
from torch.utils import data
from torchvision import transforms


def list_directory(dirname: str):
    return list(chain(*[list(Path(dirname).rglob("*." + ext)) for ext in ["png", "jpg", "jpeg", "JPG"]]))


class DefaultDataset(data.Dataset):
    def __init__(self, dirname: str, transform: transforms.Compose = None) -> None:
        self.samples = list_directory(dirname)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index: int) -> torch.Tensor:
        filename = self.samples[index]
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def get_eval_loader(
    img_folder: str, img_size: int = 256, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, drop_last: bool = False
) -> data.DataLoader:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    return data.DataLoader(
        dataset=DefaultDataset(img_folder, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class InceptionV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        self.block4 = nn.Sequential(inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = InceptionV3()
    inception.eval()
    inception.to(device)

    loaders = [get_eval_loader(path, img_size, batch_size, num_workers=4) for path in paths]
    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, desc="Calculating FID", leave=False):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, nargs=2, help="paths to real and fake images")
    parser.add_argument("--img_size", type=int, default=256, help="image resolution")
    parser.add_argument("--batch_size", type=int, default=9, help="batch size to use")
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size)
    print(f"FID: {fid_value}")
