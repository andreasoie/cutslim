import os
import random
import shutil
import warnings
from collections import OrderedDict
from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np
import torch
from matplotlib import pyplot as plt
from munch import Munch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from config import load_cfg, save_cfg
from dataset import UnalignedDataset
from fid import calculate_fid_given_paths
from generate import generate_cherry_images
from grid import create_img_grid
from model import CycleGANModel
from utils import find_epochs, save_images

torch.backends.cudnn.benchmark = True
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

warnings.filterwarnings("ignore")
os.makedirs("snapshots", exist_ok=True)
os.makedirs("cherries", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


@torch.no_grad()
def save_cherry_image(opt: Munch, model: CycleGANModel, filename: str) -> None:
    args = deepcopy(opt)
    args.dataroot = "/home/andreoi/data/study_cases"
    args.phase = "test"
    args.isTrain = False
    args.batch_size = 1
    args.shuffle = False
    args.num_threads = 4

    dataloader_val = DataLoader(
        dataset=UnalignedDataset(args), batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_threads, drop_last=False
    )
    assert len(dataloader_val) == 6, "Only 6 are expected"

    generated_images = []
    for inputs in dataloader_val:
        outputs = model.generate_visuals_for_evaluation(inputs, mode="forward")
        outputs["fake_B"] = outputs["fake_B"].squeeze(0)
        generated_images.append(outputs["fake_B"])

    img_grid = make_grid(generated_images, nrow=len(generated_images), padding=0, pad_value=1)
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    img_grid = (img_grid * 0.5) + 0.5
    img_grid = np.clip(img_grid, 0, 1)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(img_grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()


def save_snapshot_image(visuals: OrderedDict, filename: str) -> None:
    randidx = 0  # 1 out of BS
    generated_images = []
    real_A = visuals["real_A"][randidx]
    real_B = visuals["real_B"][randidx]
    fake_B = visuals["fake_B"][randidx]
    generated_images = [real_A, fake_B, real_B]
    generated_images = [img.repeat(3, 1, 1) if img.shape[0] == 1 else img for img in generated_images]
    generated_images = [img.squeeze(0) for img in generated_images]

    img_grid = make_grid(generated_images, nrow=len(generated_images), padding=0, pad_value=1)
    img_grid = img_grid.cpu().numpy().transpose((1, 2, 0))
    img_grid = (img_grid * 0.5) + 0.5
    img_grid = np.clip(img_grid, 0, 1)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(img_grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()


@torch.no_grad()
def generate_images(model: CycleGANModel, dataloader, savedir) -> float:
    for data in dataloader:
        model.set_input(data)
        model.run_test()
        visuals = model.get_current_visuals()
        paths = model.get_image_paths()
        save_images(savedir, visuals, paths)


@torch.no_grad()
def calculate_metric(opt: Munch, model: CycleGANModel) -> float:
    args = deepcopy(opt)
    args.phase = "test"
    args.isTrain = False

    evaldataset = UnalignedDataset(opt)
    evaldataloader = DataLoader(evaldataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_threads)

    with TemporaryDirectory() as tempdir:
        generate_images(model, evaldataloader, tempdir)
        paths = [opt.dataroot + "/testB", tempdir]
        return calculate_fid_given_paths(paths=paths, img_size=opt.load_size, batch_size=opt.batch_size_fid)


if __name__ == "__main__":
    baseline = load_cfg("baseline.json")
    opt = deepcopy(baseline)
    # Default parameters
    params = Munch()
    params.batch_size = 24
    params.num_threads = 12
    params.gpu_ids = [0, 1, 2]
    params.lr = 0.0006

    # Tuning parameters
    params.lr = 0.0003

    # Update opt
    opt.update(params)

    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_threads)

    model = CycleGANModel(opt)
    model.setup()
    model.trainmode()

    if opt.wandb:
        wandb.init(project="cyc", entity="andreasoie")
        # Add baseline + hyperparams
        wandb.config.update(opt)
        # Save configurations
        save_baseline = "params/baseline.json"
        save_params = "params/params.json"
        save_cfg(save_baseline, baseline)
        save_cfg(save_params, params)
        wandb.save(save_baseline)
        wandb.save(save_params)

    iterations = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        current_lr = model.optimizers[0].param_groups[0]["lr"]
        desc = f"Epoch {(epoch):3d} / {(opt.n_epochs + opt.n_epochs_decay):3d}, LR = {current_lr:.6f}"
        for i, data in tqdm(enumerate(dataloader), total=(len(dataset) // opt.batch_size), desc=desc, colour="cyan"):
            iterations += data["A"].size(0)

            model.set_input(data)
            model.optimize_parameters()

            if iterations % opt.freq_display == 0:
                filename1 = os.path.join("snapshots", f"{epoch}_{iterations}.png")
                filename2 = os.path.join("cherries", f"{epoch}_{iterations}.png")
                model.evalmode()
                save_snapshot_image(model.get_current_visuals(), filename1)
                save_cherry_image(opt, model=model, filename=filename2)
                model.trainmode()
                if opt.wandb:
                    wandb.log({"snapshot": wandb.Image(filename1)})
                    wandb.log({"cherry": wandb.Image(filename2)})

            if iterations % opt.freq_log == 0:
                losses = model.get_current_losses()
                meta = {"epoch": epoch, "iteration": iterations, **losses}
                if opt.wandb:
                    wandb.log(meta)

        if epoch % opt.freq_save_epoch == 0:
            model.evalmode()
            fid = calculate_metric(opt, model)
            model.trainmode()
            if opt.wandb:
                wandb.log({"fid": fid})
            for save_path in model.save_networks(epoch):
                pass  # don't track save_path's

        model.update_learning_rate()

    # Evaluation

    with TemporaryDirectory() as tempdir:
        epochs = find_epochs(opt.checkpoints_dir + "/" + opt.name)

        print("Generating cherry images...")
        generate_cherry_images(epochs, tempdir)

        print("Creating cherry collection...")
        gridpath = create_img_grid(epochs, tempdir, "cherry_grid.png")

        if gridpath is None:
            print("No cherry images found!")
        else:
            if opt.wandb:
                wandb.log({"img_grid": wandb.Image(gridpath)})
            else:
                os.makedirs("cherry_grids", exist_ok=True)
                shutil.copy(gridpath, "cherry_grids")
