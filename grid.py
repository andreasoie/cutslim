import os
import time
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

N_OBJECTS = 6


def find_files(dirpath: str, suffix: str) -> List[str]:
    return sorted([os.path.join(rootdir, f) for rootdir, _, files in os.walk(dirpath) for f in files if f.lower().endswith(f".{suffix}")])


def create_img_grid(epochs: List[int], tmpdir: str, figurename: str) -> Union[None, str]:
    DOMAIN_A_IMAGES = "/home/andreoi/data/study_cases/testA"
    DOMAIN_B_IMAGES = "/home/andreoi/data/study_cases/testB"

    eo_images = find_files(DOMAIN_A_IMAGES, "png")
    ir_images = find_files(DOMAIN_B_IMAGES, "png")
    eo_images = sorted(eo_images)
    ir_images = sorted(ir_images)

    imgfolder = "outputs"
    image_collection = []

    # Add the first image from each domain
    for img in eo_images:
        image_collection.append(img)

    if len(epochs) < 1:
        return None

    for resume_epoch in epochs:
        # result_path = f"/home/andreoi/dev/{repo}/{imgfolder}/{resume_epoch}"
        result_path = f"{tmpdir}/{resume_epoch}"

        assert os.path.exists(result_path), f"Path {result_path} does not exist"
        result_images = find_files(result_path, "png")
        result_images = sorted(result_images)
        for ri in result_images:
            image_collection.append(ri)

    for img in ir_images:
        image_collection.append(img)

    accumulated_size = 0
    tensor_list = []
    for img_file in image_collection:
        accumulated_size += 1
        img_data = cv2.imread(img_file)
        img_data = cv2.resize(img_data, (256, 256))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0  # Convert to PyTorch tensor
        img_data = torch.nn.functional.interpolate(img_data.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
        tensor_list.append(img_data)

    grid = vutils.make_grid(tensor_list, nrow=N_OBJECTS, padding=0, pad_value=1)
    grid = grid.numpy().transpose((1, 2, 0))
    fig, ax = plt.subplots(figsize=(15, 20))
    ax.imshow(grid)
    ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    saveas = os.path.join(tmpdir, figurename)
    plt.savefig(saveas, dpi=300, transparent=True)

    time.sleep(5)
    if os.path.exists(imgfolder):
        os.system(f"rm -rf {imgfolder}")

    return saveas
