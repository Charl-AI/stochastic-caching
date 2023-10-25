import ctypes
import multiprocessing as mp
from typing import Literal

import numpy as np
import torch

# simply maps dtype names to ctypes types and bit widths
CTYPE_MAP = {
    "8-bit": (ctypes.c_uint8, 8),
    "16-bit": (ctypes.c_uint16, 16),
    "32-bit": (ctypes.c_uint32, 32),
    "64-bit": (ctypes.c_uint64, 64),
}


class SharedCache:
    """A simple shared memory cache for use in PyTorch datasets.

    You can set a size limit for the cache to take. If your dataset
    exceeds this size, the cache will only allocate space for the first N samples.
    This allows you to speed up training by caching only a subset of your dataset.
    When applied this to a large, shuffled dataset, we call this 'stochastic caching'.

    N.B. It is recommended to set the size limit to <= the size of
    /dev/shm. You may check this with `df -h`, and temporarily increase
    it with `mount -o remount,size=128G /dev/shm` (for 128 GiB, for example).

    This library includes a helper function `get_shm_size()` for getting the size of
    /dev/shm in python.

    Example usage:

    ```python
    from stocaching import SharedCache
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self):
            super().__init__()

            ... # set up dataset

            dataset_len = 10000
            data_dims = (3, 32, 32)

            # initialize the cache
            self.cache = SharedCache(
                size_limit_gib=32,
                dataset_len=dataset_len,
                data_dims=data_dims,
                dtype="8-bit",
            )
        def __getitem__(self, idx):
            # check if item is in cache
            if not self.cache.is_slot_empty(idx):
                img = self.cache[idx]
            else:
                img = ... # load uint8 tensor
                self.cache.set_slot(idx, img)
            return img
    ```
    """

    def __init__(
        self,
        size_limit_gib: int,
        dataset_len: int,
        data_dims: tuple[int, ...],
        dtype: Literal["8-bit", "16-bit", "32-bit", "64-bit"] = "8-bit",
    ) -> None:
        """
        Args:
            size_limit_gib (int): Maximum size of the cache in GiB
            dataset_len (int): Length (number of samples) of the full dataset.
            data_dims (tuple[int, ...]): Dimensions of the data to be stored in the
                cache. E.g. (C, H, W) for 2D image data. Do not include batch dim.
            dtype (Literal["8-bit", "16-bit", "32-bit", "64-bit"], optional): Data type
                of the dataset. 8-bit (i.e. 0-255 int imgs) are the recommended format.
                 Defaults to "8-bit".
        """
        bytes_per_gib = 1024**3

        self.lock = mp.Lock()

        data_bytes = np.prod(data_dims) * dataset_len * CTYPE_MAP[dtype][1] / 8
        size_limit_bytes = size_limit_gib * bytes_per_gib

        if data_bytes > size_limit_bytes:
            cache_len = int(
                size_limit_bytes / (np.prod(data_dims) * CTYPE_MAP[dtype][1] / 8)
            )
            print(
                f"Dataset size ({data_bytes / bytes_per_gib:.1f} GiB) exceeds cache"
                + f" limit ({size_limit_gib} GiB)."
                + f" Allocating space to cache {cache_len} / {dataset_len} samples."
            )

        else:
            cache_len = dataset_len
            print(
                f"Dataset size ({data_bytes / bytes_per_gib:.1f} GiB) fits in cache"
                + f" limit ({size_limit_gib} GiB)."
                + f" Allocating space to cache all {cache_len} samples."
            )

        shared_array_base = mp.Array(
            CTYPE_MAP[dtype][0], int(np.prod(data_dims) * cache_len)
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape((cache_len, *data_dims))
        self.shm = torch.from_numpy(shared_array)
        self.shm *= 0

    def __getitem__(self, idx: int):
        with self.lock:
            return self.shm[idx]

    def __setitem__(self, idx: int, value: torch.Tensor):
        with self.lock:
            self.shm[idx] = value

    def __len__(self):
        return len(self.shm)

    def set_slot(
        self, idx: int, value: torch.Tensor, allow_overwrite: bool = False
    ) -> None:
        """Set a slot (i.e. datapoint) in the cache to a value.
        Is a no-op if given an out-of-bounds index (just returns
        gracefully).

        Args:
            idx (int): Index of the slot (datapoint) to set.
            value (torch.Tensor): Value to set the slot to.
            allow_overwrite (bool, optional): If False, raises an error if
                the slot has any non-zero elements. Defaults to False.
        """
        if idx >= len(self) or idx < 0:
            return
        if not allow_overwrite and not self.is_slot_empty(idx):
            raise RuntimeError(
                f"Attempted to overwrite non-empty slot {idx} in SharedCache."
            )
        self[idx] = value

    def is_slot_empty(self, idx: int) -> bool:
        """Check if a slot (i.e. datapoint) in the cache has anything stored in it.
        Gracefully handles out-of-bounds indices.

        Relies on the fact that the cache is initialized to zeros
        (i.e. returns True if all(cache[idx] == 0)). Returns False
        if any non-zero values, OR if the slot is out of bounds.

        Warning: this will fail if you have any legitimate all-zero
        datapoints!

        Args:
            idx (int): Index of the slot (datapoint) to check.

        Returns:
            (bool) True if slot is empty, False otherwise.
        """
        if idx >= len(self) or idx < 0:
            return False
        return all(self.shm[idx] == 0)
