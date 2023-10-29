"""Stocaching, a tiny library for stochastic dataset caching in PyTorch."""

import ctypes
import multiprocessing as mp
import os
from enum import Enum

import numpy as np
import torch

__all__ = ["SharedCache", "get_shm_size"]

BYTES_PER_GIB = 1024**3

C_DTYPES = {
    torch.bool: ctypes.c_bool,
    torch.uint8: ctypes.c_uint8,
    torch.int8: ctypes.c_int8,
    torch.int16: ctypes.c_int16,
    torch.int32: ctypes.c_int32,
    torch.int64: ctypes.c_int64,
    torch.float32: ctypes.c_float,
    torch.float64: ctypes.c_double,
}


class SlotState(Enum):
    EMPTY = 0
    SET = 1
    # in this context, OOB means outside the range of the cache,
    # but inside the range of the full dataset
    OOB = 2


class SharedCache:
    """A simple shared memory cache for use in PyTorch datasets.

    You can set a size limit for the cache to take. If your dataset
    exceeds this size, the cache will only allocate slots for the first N samples.
    This allows you to speed up training by caching only a subset of your dataset.
    When applied to a large, shuffled dataset, we call this 'stochastic caching'.

    You may interact with the cache directly as if it were a list of slots,
    with one slot per sample. Get and set with `x = cache[0]` and `cache[0] = x`.

    Using the getter and setter directly can be fiddly if you are only caching part
    of the dataset. We expose two convenience methods (`get_slot` and `set_slot`),
    which simplify usage by allowing you to treat the cache as if it were the same
    size as the full dataset.

    You may access the underlying pytorch array with the `underlying_array property`.

    Example usage:

    ```python
    import torch
    from stocaching import SharedCache
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self):
            super().__init__()

            ... # set up dataset

            dataset_len = N   # number of samples in the full dataset
            data_dims = (3, 32, 32)   # data dims (not including batch)

            # initialize the cache
            self.cache = SharedCache(
                size_limit_gib=32,
                dataset_len=dataset_len,
                data_dims=data_dims,
                dtype=torch.uint8
            )
        def __getitem__(self, idx):
            # retrieve data from cache if it's there
            x = self.cache.get_slot(idx)
            # x will be None if the cache slot was empty or OOB
            if x is None:
                x = ... # load data to uint8 tensor from disk
                # try to cache x, no-op if idx is OOB of the cache
                self.cache.set_slot(idx, x)
            return x
    ```
    """

    def __init__(
        self,
        size_limit_gib: int,
        dataset_len: int,
        data_dims: tuple[int, ...],
        dtype: torch.dtype = torch.uint8,
    ) -> None:
        """
        Args:
            size_limit_gib (int): Maximum size of the cache in GiB.
            dataset_len (int): Length (number of samples) of the full dataset.
            data_dims (tuple[int, ...]): Dimensions of the data to be stored in the
                cache. E.g. (C, H, W) for 2D image data. Does not include batch dim.
            dtype (torch.dtype, optional): Torch data type of the dataset to cache.
                Must be in the subset of torch dtypes with corresponding ctypes:
                bool, uint8, int8, int16, int32, int64, float32, float64.
                Defaults to torch.uint8 (this is usually best for jpg images).
        """
        dtype_bytes = dtype.itemsize
        slot_bytes = np.prod(data_dims) * dtype_bytes
        dataset_bytes = slot_bytes * dataset_len
        size_limit_bytes = size_limit_gib * BYTES_PER_GIB

        # we allocate a flat 8-bit array to keep track of which samples are cached,
        # which are not cached yet, and which are out of bounds of the cache
        aux_bytes = dataset_len * torch.uint8.itemsize

        ds_and_aux_bytes = dataset_bytes + aux_bytes

        if ds_and_aux_bytes > size_limit_bytes:
            cache_len = int((size_limit_bytes - aux_bytes) / slot_bytes)
            print(
                f"Dataset size ({ds_and_aux_bytes / BYTES_PER_GIB:.1f} GiB)"
                + f" exceeds cache limit ({size_limit_gib} GiB)."
                + f" Allocating space to cache {cache_len} / {dataset_len} samples."
            )

        else:
            cache_len = dataset_len
            print(
                f"Dataset size ({ds_and_aux_bytes / BYTES_PER_GIB:.1f} GiB)"
                + f" fits in cache limit ({size_limit_gib} GiB)."
                + f" Allocating space to cache all {cache_len} samples."
            )

        shared_array_base = mp.Array(
            C_DTYPES[dtype], int(np.prod(data_dims) * cache_len)
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape((cache_len, *data_dims))
        self._shm = torch.from_numpy(shared_array)
        self._shm *= 0

        shared_aux_base = mp.Array(C_DTYPES[torch.uint8], dataset_len)
        shared_aux = np.ctypeslib.as_array(shared_aux_base.get_obj())
        self._aux = torch.from_numpy(shared_aux)
        self._aux *= 0

        # only cache the first cache_len samples by index
        self._aux[cache_len:] = SlotState.OOB.value

    @property
    def array(self) -> torch.Tensor:
        """Access the full underlying cache (just a tensor backed by shared memory).
        Returns a torch tensor of shape (cache_len, *data_dims).
        The dtype is whatever you specified when constructing the cache."""
        return self._shm

    @property
    def aux_array(self) -> torch.Tensor:
        """Access the auxiliary array (just a tensor backed by shared memory).
        The auxiliary array keeps track of which samples from the full dataset have been
        cached, which samples are yet to be cached, and which are OOB.
        Returns a shared memory torch uint8 tensor, shape (dataset_len,).
        `self.aux_array[idx] == 0` means sample idx is not cached.
        `self.aux_array[idx] == 1` means sample idx is cached.
        `self.aux_array[idx] == 2` means sample idx is OOB.
        """
        return self._aux

    def __getitem__(self, idx: int):
        return self.array[idx]

    def __setitem__(self, idx: int, value: torch.Tensor):
        self.array[idx] = value

    def __len__(self):
        return len(self.array)

    def _slot_state(self, idx: int) -> SlotState:
        """Get the state of a slot in the cache. Raises an error if idx is outside
        the range of the full dataset."""
        if idx < 0 or idx >= len(self.aux_array):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self.aux_array)}"
            )
        return SlotState(self._aux[idx].item())

    def set_slot(
        self,
        idx: int,
        value: torch.Tensor,
        allow_oob_idx: bool = True,
        allow_overwrite: bool = False,
    ) -> None:
        """Set a slot in the cache to a value.

        The main reason to use this method over __setitem__ is that we
        allow you to call this method when idx is out of bounds of the
        cache, but within the range of the full dataset.

        In this case the method is a no-op when idx is out of bounds.

        Args:
            idx (int): Index of the slot to set.
            value (torch.Tensor): Value to set the slot to.
            allow_oob_idx (bool, optional): When False, raises an error if
                idx is out of bounds of the cache. Defaults to True.
            allow_overwrite (bool, optional): When False, raises an error if
                the slot has any existing non-zero elements. Defaults to False.
        """
        slot_state = self._slot_state(idx)
        if slot_state == SlotState.OOB:
            if not allow_oob_idx:
                raise IndexError(
                    f"Index {idx} out of bounds of SharedCache of length {len(self)}"
                )
            return  # no-op

        if slot_state == SlotState.SET and not allow_overwrite:
            raise RuntimeError(
                f"Tried to overwrite non-empty slot {idx=} in SharedCache."
            )

        self[idx] = value
        self.aux_array[idx] = SlotState.SET.value

    def get_slot(
        self,
        idx: int,
        allow_oob_idx: bool = True,
        allow_empty_slot: bool = True,
    ) -> torch.Tensor | None:
        """Get the value of a slot in the cache.

        The main reason to use this method over __getitem__ is that we
        allow you to call this method when idx is out of bounds of the
        cache, but within the range of the dataset.

        In this case the method returns None when idx is out of bounds.

        Args:
            idx (int): Index of the slot to get.
            allow_oob_idx (bool, optional): When False, raises an error if
                idx is out of bounds of the cache. Defaults to True.
            allow_empty_slot (bool, optional): When True, returns
                None if the slot is empty. Otherwise, raises
                 an exception. Defaults to True.
        """
        slot_state = self._slot_state(idx)
        if slot_state == SlotState.OOB:
            if not allow_oob_idx:
                raise IndexError(
                    f"Index {idx} out of bounds of SharedCache of length {len(self)}"
                )
            return None

        if slot_state == SlotState.EMPTY:
            if allow_empty_slot:
                return None
            else:
                raise RuntimeError(
                    f"Tried to read from an empty slot {idx=} in SharedCache."
                )
        return self[idx]

    def clear(self) -> None:
        """Clear all slots in the cache."""
        self._shm *= 0
        self._aux *= 0
        self._aux[len(self) :] = SlotState.OOB.value


def get_shm_size() -> int:
    """Get size of /dev/shm. The size limit of the shared memory cache
    should not exceed this.

    N.B. You may check the size of /dev/shm on the command line with `df -h`.
    A simple way to (temporarily) change it is to run:
    `mount -o remount,size=128G /dev/shm` (change to 128 GiB, for example).

    Returns:
        (int) Size of /dev/shm in GiB
    """
    stats = os.statvfs("/dev/shm")
    shm_bytes = stats.f_bsize * stats.f_blocks
    shm_size = shm_bytes / BYTES_PER_GIB
    return int(shm_size)
