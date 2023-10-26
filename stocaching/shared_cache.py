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
    exceeds this size, the cache will only allocate slots for the first N samples.
    This allows you to speed up training by caching only a subset of your dataset.
    When applied to a large, shuffled dataset, we call this 'stochastic caching'.

    N.B. It is recommended to set the size limit to <= the size of
    /dev/shm. You may check this with `df -h`, and temporarily increase
    it with `mount -o remount,size=128G /dev/shm` (for 128 GiB, for example).

    Once initialised, you may interact with the cache directly as if it were a list
    of slots, with each slot holding a datapoint: i.e. `x = cache[0]` and `cache[0] = x`

    We also provide two convenience methods (`is_slot_empty` and `set_slot`) for
    simplifying common use cases.

    Example usage:

    ```python
    from stocaching import SharedCache
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self):
            super().__init__()

            ... # set up dataset

            dataset_len = N   # number of samples in the full dataset
            data_dims = (C, H, W)   # img dims (not including batch)

            # initialize the cache
            self.cache = SharedCache(
                size_limit_gib=32,
                dataset_len=dataset_len,
                data_dims=data_dims,
                dtype="8-bit",
            )
        def __getitem__(self, idx):
            # retrieve img from cache if it's there
            if not self.cache.is_slot_empty(idx):
                img = self.cache[idx]
            # otherwise, load it from disk and try to cache it
            else:
                img = ... # load uint8 tensor from disk
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
            size_limit_gib (int): Maximum size of the cache in GiB.
            dataset_len (int): Length (number of samples) in the full dataset.
            data_dims (tuple[int, ...]): Dimensions of the data to be stored in the
                cache. E.g. (C, H, W) for 2D image data. Do not include batch dim.
            dtype (Literal["8-bit", "16-bit", "32-bit", "64-bit"], optional): Data type
                of the dataset to cache. 8-bit (i.e. 0-255 int) is the recommended
                format. Defaults to "8-bit".
        """
        bytes_per_gib = 1024**3

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

        # separate locks for each slot
        self.locks = [mp.Lock() for _ in range(cache_len)]

    def __getitem__(self, idx: int):
        with self.locks[idx]:
            return self.shm[idx]

    def __setitem__(self, idx: int, value: torch.Tensor):
        with self.locks[idx]:
            self.shm[idx] = value

    def __len__(self):
        return len(self.shm)

    def is_slot_empty(self, idx: int, allow_oob_idx: bool = True) -> bool:
        """Check if a slot (i.e. datapoint) in the cache has anything stored in it.
        Can optionally handle out-of-bounds indices gracefully.

        Relies on the fact that the cache is initialized to zeros
        (i.e. returns True if all(cache[idx] == 0)). Returns False
        if any non-zero values, OR if the slot is out of bounds.

        Warning: this will fail if you have any legitimate all-zero
        datapoints!

        Args:
            idx (int): Index of the slot (datapoint) to check.
            allow_oob_idx (bool, optional): If False, raises an error if
                the index is out of bounds. Defaults to True.
        Returns:
            (bool) True if slot is empty, False otherwise.
        """
        if idx >= len(self) or idx < 0:
            if allow_oob_idx:
                return False
            else:
                raise IndexError(f"Index {idx} is out of bounds for SharedCache.")
        return bool(torch.all(self.shm[idx] == 0))

    def set_slot(
        self,
        idx: int,
        value: torch.Tensor,
        allow_overwrite: bool = False,
        allow_oob_idx: bool = True,
    ) -> None:
        """Set a slot (i.e. datapoint) in the cache to a value.
        Can optionally handle out-of-bounds indices gracefully.

        Args:
            idx (int): Index of the slot (datapoint) to set.
            value (torch.Tensor): Value to set the slot to.
            allow_overwrite (bool, optional): If False, raises an error if
                the slot has any existing non-zero elements. Defaults to False.
            allow_oob_idx (bool, optional): If False, raises an error if
                the index is out of bounds. Else returns gracefully
                 as a no-op. Defaults to True.
        """
        if idx >= len(self) or idx < 0:
            if allow_oob_idx:
                return
            else:
                raise IndexError(f"Index {idx} is out of bounds for SharedCache.")

        if not allow_overwrite and not self.is_slot_empty(idx):
            raise RuntimeError(
                f"Tried to overwrite non-empty slot {idx=} in SharedCache."
            )

        self[idx] = value
