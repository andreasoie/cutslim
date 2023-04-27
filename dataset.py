import os.path
import random

from PIL import Image

import transforms

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    assert os.path.isdir(dir) or os.path.islink(dir), f"{dir} is not a valid directory"
    return [os.path.join(root, fname) for root, _, fnames in sorted(os.walk(dir, followlinks=True)) for fname in fnames if is_image_file(fname)]


class UnalignedDataset:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        if opt.phase == "test" and not os.path.exists(self.dir_A) and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.opt.phase == "train":
            self.transformA = transforms.get_transforms_training(self.opt, grayscale=True if self.opt.input_nc == 1 else False)
            self.transformB = transforms.get_transforms_training(self.opt, grayscale=True if self.opt.output_nc == 1 else False)
        else:
            self.transformA = transforms.get_transforms_testing(self.opt, grayscale=True if self.opt.input_nc == 1 else False)
            self.transformB = transforms.get_transforms_testing(self.opt, grayscale=True if self.opt.output_nc == 1 else False)

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index: int) -> dict:
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        A_img = self.transformA(Image.open(A_path).convert("RGB"))
        B_img = self.transformB(Image.open(B_path).convert("RGB"))
        return {"A": A_img, "B": B_img, "A_paths": A_path, "B_paths": B_path}
