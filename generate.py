import argparse
import os
import random
import warnings
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_cfg
from dataset import UnalignedDataset
from model import CycleGANModel
from utils import find_epochs, save_images

torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

warnings.filterwarnings("ignore")


def generate_images(epoch: int, model: CycleGANModel, dataloader: DataLoader, savedir: str) -> float:
    for data in tqdm(dataloader, desc=f"{str(epoch).ljust(3)} Generating"):
        model.set_input(data)
        model.run_test()
        visuals = model.get_current_visuals()
        paths = model.get_image_paths()
        save_images(savedir, visuals, paths)


def generate_cherry_images(epochs: List[int], savedir: str) -> None:
    opt = load_cfg("baseline.json")
    opt.dataroot = "/home/andreoi/data/study_cases"
    opt.phase = "test"
    opt.isTrain = False
    opt.shuffle = False
    opt.drop_last = False
    opt.batch_size = 1
    opt.num_threads = 4
    opt.gpu_ids = [0]

    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_threads, drop_last=opt.drop_last)

    for resume_epoch in epochs:
        opt.epoch = resume_epoch
        model = CycleGANModel(opt)
        model.setup()
        model.evalmode()

        tempdir = f"{savedir}/{resume_epoch}"
        os.makedirs(tempdir, exist_ok=True)
        generate_images(resume_epoch, model, dataloader, tempdir)
