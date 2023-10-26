import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import FakeData
from torchvision.transforms import v2
from tqdm import tqdm

from stocaching import SharedCache

DATASET_LEN = 50000
RAW_DATA_DIMS = (3, 512, 512)
CACHED_DATA_DIMS = (3, 256, 256)
FINAL_DATA_DIMS = (3, 224, 224)

# for normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_dummy_data(
    data_dir: str,
    size: int = DATASET_LEN,
    img_size: tuple[int, int, int] = RAW_DATA_DIMS,
) -> None:
    print(f"Saving dummy data to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    ds = FakeData(size, img_size)
    skipped = 0
    for i in tqdm(range(len(ds)), desc="Saving dummy data"):
        img_path = os.path.join(data_dir, f"{i}.jpg")
        if os.path.exists(img_path):
            skipped += 1
            continue
        x, _ = ds[i]
        img: Image.Image = x
        img.save(img_path)

    print(
        f"Dummy data saved to {data_dir}"
        + f" ({size - skipped} / {size} images saved. {skipped} imgs already existed)."
    )


def get_transforms() -> Callable[[Image.Image], torch.Tensor]:
    """Transforms map a PIL image to a uint8 torch Tensor.
    This is applied before caching, so it is important that no stochastic
    operations are included.
    """
    transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8),
        v2.Resize(CACHED_DATA_DIMS[1], antialias=True),
    ]

    return v2.Compose(transform_list)


def get_augmentations() -> Callable[[torch.Tensor], torch.Tensor]:
    """Augmentations map a torch uint8 Tensor to a normalised float Tensor
    This is applied after caching, so should include all stochastic operations.
    """
    aug_list = [
        v2.RandomResizedCrop(FINAL_DATA_DIMS[1], antialias=True),
        v2.RandAugment(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(aug_list)


class DummyDataset(Dataset):
    def __init__(self, data_dir: str, cache_limit_gib: int):
        """PyTorch dataset for dummy data.
        No cache is used if cache_limit_gib is 0."""
        self.data_dir = data_dir
        self.cache_limit_gib = cache_limit_gib
        self.transforms = get_transforms()
        self.augmentations = get_augmentations()

        save_dummy_data(data_dir)

        if cache_limit_gib != 0:
            self.cache = SharedCache(
                cache_limit_gib, DATASET_LEN, CACHED_DATA_DIMS, dtype="8-bit"
            )

    def _get_img(self, idx) -> torch.Tensor:
        """Reads dummy data from disk to a uint8 torch tensor."""
        img_path = os.path.join(self.data_dir, f"{idx}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return DATASET_LEN

    def __getitem__(self, idx) -> torch.Tensor:
        # caching disabled
        if self.cache_limit_gib == 0:
            return self.augmentations(self._get_img(idx))

        # check if img is in cache
        if not self.cache.is_slot_empty(idx):
            img = self.cache[idx]
        # otherwise, read from disk and try to cache
        else:
            img = self._get_img(idx)  # uint8 tensor
            self.cache.set_slot(idx, img)

        return self.augmentations(img)
