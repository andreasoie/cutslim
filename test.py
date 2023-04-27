import random
import warnings
from tempfile import TemporaryDirectory

import numpy as np
import torch
from halo import Halo
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_cfg
from dataset import UnalignedDataset
from fid import calculate_fid_given_paths
from model import CycleGANModel
from utils import find_epochs, save_images

torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

warnings.filterwarnings("ignore")


def generate_images(epoch: int, model: CycleGANModel, dataloader, savedir) -> float:
    for data in tqdm(dataloader, desc=f"{str(epoch).ljust(3)} Generating"):
        model.set_input(data)
        model.run_test()
        visuals = model.get_current_visuals()
        paths = model.get_image_paths()
        save_images(savedir, visuals, paths)


if __name__ == "__main__":
    opt = load_cfg("baseline.json")
    opt.results_dir = "outputs"
    opt.batch_size = 24
    opt.batch_size_fid = 9
    opt.num_threads = 12
    opt.gpu_ids = [0, 1, 2]

    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_threads)

    for resume_epoch in find_epochs(opt.checkpoints_dir + "/" + opt.name):  # opt.checkpoints_dir + "/" + opt.name
        opt.epoch = resume_epoch
        model = CycleGANModel(opt)
        model.setup()
        model.evalmode()

        with TemporaryDirectory() as tempdir:
            generate_images(resume_epoch, model, dataloader, tempdir)
            paths = [opt.dataroot + "/testB", tempdir]
            fid = calculate_fid_given_paths(paths=paths, img_size=opt.load_size, batch_size=opt.batch_size_fid)
