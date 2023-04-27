import argparse
import json
import os

from munch import Munch

import utils


def load_cfg(name: str):
    workdir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{workdir}/configs/{name.lower()}", "r") as f:
        return Munch(json.load(f))


def save_cfg(saveas: str, cfg: Munch):
    with open(saveas, "w") as f:
        json.dump(cfg, f, indent=4)


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        # basic
        self.parser.add_argument("--dataroot", default="placeholder", help="path to images (should have subfolders trainA, trainB, valA, valB, etc)")
        self.parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment")
        self.parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        self.parser.add_argument("--phase", type=str, default="train", help="train, test")
        # model
        self.parser.add_argument("--model", type=str, default="cut", help="chooses which model to use.")
        self.parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels: 3 for RGB and 1 for grayscale")
        self.parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels: 3 for RGB and 1 for grayscale")
        self.parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in the last conv layer")
        self.parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in the first conv layer")
        self.parser.add_argument("--netD", type=str, default="basic", choices=["basic", "n_layers", "pixel", "patch", "tilestylegan2", "stylegan2"])
        self.parser.add_argument(
            "--netG",
            type=str,
            default="resnet_9blocks",
            choices=["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128", "resnet_cat"],
        )
        self.parser.add_argument("--n_layers_D", type=int, default=3, help="only used if netD==n_layers")
        self.parser.add_argument(
            "--normG", type=str, default="instance", choices=["instance", "batch", "none"], help="instance normalization or batch normalization for G"
        )
        self.parser.add_argument(
            "--normD", type=str, default="instance", choices=["instance", "batch", "none"], help="instance normalization or batch normalization for D"
        )
        self.parser.add_argument(
            "--init_type", type=str, default="xavier", choices=["normal", "xavier", "kaiming", "orthogonal"], help="network initialization"
        )
        self.parser.add_argument("--init_gain", type=float, default=0.02, help="scaling factor for normal, xavier and orthogonal.")
        self.parser.add_argument("--no_dropout", type=utils.str2bool, nargs="?", const=True, default=True, help="no dropout for the generator")
        self.parser.add_argument(
            "--no_antialias", action="store_true", help="if specified, use stride=2 convs instead of antialiased-downsampling (sad)"
        )
        self.parser.add_argument(
            "--no_antialias_up",
            action="store_true",
            help="if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]",
        )
        self.parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
        self.parser.add_argument("--num_threads", default=4, type=int, help="# threads for loading data")
        self.parser.add_argument("--batch_size", type=int, default=8, help="input batch size")
        self.parser.add_argument("--load_size", type=int, default=256, help="scale images to this size")
        self.parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
        self.parser.add_argument("--epoch", type=int, default=0, help="which epoch to load?")
        self.parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")

    def parse(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        return args


class TrainConfig(BaseConfig):
    def initialize(self) -> argparse.ArgumentParser:
        super().initialize()
        # network saving and loading parameters
        self.parser.add_argument("--continue_train", action="store_true", help="continue training: load the latest model")
        self.parser.add_argument("--epoch_count", type=int, default=1, help="the starting epoch count,")
        self.parser.add_argument("--pretrained_name", type=str, default=None, help="resume training from another checkpoint")
        self.parser.add_argument("--shuffle", type=bool, default=True, help="shuffle input data")
        self.parser.add_argument("--drop_last", type=bool, default=False, help="drop last batch")

        # parameters
        self.parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs with the initial learning rate")
        self.parser.add_argument("--n_epochs_decay", type=int, default=200, help="number of epochs to linearly decay learning rate to zero")
        self.parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="momentum term of adam")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
        self.parser.add_argument("--gan_mode", type=str, default="lsgan", help="the type of GAN objective. [vanilla| lsgan | wgangp]")
        self.parser.add_argument("--pool_size", type=int, default=50, help="the size of image buffer that stores previously generated images")
        self.parser.add_argument("--lr_policy", type=str, default="linear", help="learning rate policy. [linear | step | plateau | cosine]")
        self.parser.add_argument("--lr_decay_iters", type=int, default=50, help="multiply by a gamma every lr_decay_iters iterations")
        self.parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
        self.parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
        self.parser.add_argument(
            "--lambda_identity",
            type=float,
            default=0.0,
            help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
        )
        # logging
        self.parser.add_argument("--freq_track", type=int, default=1000, help="track metrics every freq_track iterations")
        self.parser.add_argument("--freq_print", type=int, default=100, help="print metrics every freq_print iterations")
        self.parser.add_argument("--freq_save", type=int, default=25, help="save very freq_save epochs")
        self.parser.add_argument("--wandb", action="store_true", help="whether to use wandb")


class TestConfig(BaseConfig):
    def initialize(self) -> argparse.ArgumentParser:
        super().initialize()
        self.parser.add_argument("--outdir", type=str, default="outdir")
        self.parser.add_argument("--shuffle", type=bool, default=False, help="shuffle input data")
        self.parser.add_argument("--drop_last", type=bool, default=True, help="drop last batch")
        self.parser.add_argument("--batch_size_fid", type=int, default=9, help="batch size for FID")
