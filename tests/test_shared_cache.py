import multiprocessing as mp

import numpy as np
import pytest
import torch

from stocaching import SharedCache

BYTES_PER_GIB = 1024**3

CACHE_SIZE_GIB = 1
DATASET_SIZE_GIB = 2
DATA_DIMS = (3, 32, 32)
DTYPE = torch.uint8

DATASET_LEN = (
    DATASET_SIZE_GIB
    * BYTES_PER_GIB
    // (DATA_DIMS[0] * DATA_DIMS[1] * DATA_DIMS[2] * DTYPE.itemsize)
)


@pytest.fixture(scope="module")
def cache() -> SharedCache:
    return SharedCache(
        CACHE_SIZE_GIB,
        DATASET_LEN,
        DATA_DIMS,
        dtype=DTYPE,
    )


def test_array_property(cache: SharedCache):
    cache_array = cache.array
    assert isinstance(cache_array, torch.Tensor)
    assert cache_array.shape == (len(cache),) + DATA_DIMS
    assert cache_array.dtype == DTYPE


def test_aux_array_property(cache: SharedCache):
    aux_array = cache.aux_array
    assert isinstance(aux_array, torch.Tensor)
    assert aux_array.shape == (DATASET_LEN,)
    assert aux_array.dtype == DTYPE


def test_cache_size(cache: SharedCache):
    cache_array = cache.array
    aux_array = cache.aux_array

    cache_bytes = np.prod(cache_array.shape) * cache_array.dtype.itemsize
    aux_bytes = np.prod(aux_array.shape) * aux_array.dtype.itemsize

    assert cache_bytes + aux_bytes <= CACHE_SIZE_GIB * BYTES_PER_GIB
    assert len(cache_array) == (CACHE_SIZE_GIB * BYTES_PER_GIB - aux_bytes) // (
        np.prod(DATA_DIMS) * DTYPE.itemsize
    )


def test_set_slot(cache: SharedCache):
    # set a random slot within bounds of the cache
    cache_len = len(cache)
    idx_to_set = int(torch.randint(0, cache_len, (1,)).item())
    value = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)

    cache.set_slot(idx_to_set, value, allow_oob_idx=False, allow_overwrite=False)
    assert torch.all(cache[idx_to_set] == value)

    value2 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    cache.set_slot(idx_to_set, value2, allow_overwrite=True)
    assert torch.all(cache[idx_to_set] == value2)

    value3 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        cache.set_slot(idx_to_set, value3, allow_overwrite=False)

    # set a random slot outside the bounds of the cache
    ds_len = len(cache.aux_array)
    if ds_len != cache_len:
        idx_to_set = int(torch.randint(cache_len, ds_len, (1,)).item())
        value4 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)

        with pytest.raises(IndexError):
            cache.set_slot(idx_to_set, value4, allow_oob_idx=False)

        cache.set_slot(idx_to_set, value4, allow_oob_idx=True)

    # try to set a slot outside the bounds of the dataset
    # note how it should raise an error even if allow_oob_idx=True
    idx = ds_len + 1
    value5 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    with pytest.raises(IndexError):
        cache.set_slot(idx, value5, allow_oob_idx=True)


def test_get_slot(cache: SharedCache):
    # get a random slot within bounds of the cache
    cache_len = len(cache)
    idx_to_get = int(torch.randint(0, cache_len, (1,)).item())

    with pytest.raises(RuntimeError):
        _ = cache.get_slot(idx_to_get, allow_oob_idx=False, allow_empty_slot=False)

    val = cache.get_slot(idx_to_get, allow_oob_idx=False, allow_empty_slot=True)
    assert val is None

    set_val = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    cache.set_slot(idx_to_get, set_val, allow_overwrite=False)
    val = cache.get_slot(idx_to_get, allow_oob_idx=False, allow_empty_slot=False)
    assert torch.all(val == set_val)

    # get a random slot outside the bounds of the cache,
    # but within the bounds of the dataset
    ds_len = len(cache.aux_array)
    if ds_len != cache_len:
        idx_to_get = int(torch.randint(cache_len, ds_len, (1,)).item())
        with pytest.raises(IndexError):
            _ = cache.get_slot(idx_to_get, allow_oob_idx=False)
        val = cache.get_slot(idx_to_get, allow_oob_idx=True)
        assert val is None

    # try to get a slot outside the bounds of the dataset
    # note how it should raise an error even if allow_oob_idx=True
    idx_to_get = ds_len + 1
    with pytest.raises(IndexError):
        _ = cache.get_slot(idx_to_get)


def test_clear_slots(cache: SharedCache):
    cache_len = len(cache)
    idx_to_set = int(torch.randint(0, cache_len, (1,)).item())

    value = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    cache.set_slot(idx_to_set, value, allow_oob_idx=False, allow_overwrite=False)
    assert torch.all(cache[idx_to_set] == value)

    value2 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    with pytest.raises(RuntimeError):
        cache.set_slot(idx_to_set, value2, allow_overwrite=False)

    cache.clear()

    with pytest.raises(RuntimeError):
        _ = cache.get_slot(idx_to_set, allow_oob_idx=False, allow_empty_slot=False)

    val = cache.get_slot(idx_to_set, allow_oob_idx=False, allow_empty_slot=True)
    assert val is None

    cache.set_slot(idx_to_set, value2, allow_overwrite=False)
    assert torch.all(cache[idx_to_set] == value2)


@pytest.mark.skip(reason="Not a test, just a helper")
def set_slot(cache: SharedCache, idx: int, value: torch.Tensor):
    cache.set_slot(idx, value, allow_overwrite=True, allow_oob_idx=False)


@pytest.mark.skip(reason="Not a test, just a helper")
def get_slot(cache: SharedCache, idx: int):
    return cache.get_slot(idx, allow_oob_idx=False, allow_empty_slot=False)


def test_slots_multiprocess(cache: SharedCache):
    cache_len = len(cache)
    write_idx_1 = int(torch.randint(0, cache_len, (1,)).item())
    write_idx_2 = int(torch.randint(0, cache_len, (1,)).item())
    write_value_1 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)
    write_value_2 = torch.randint(0, 255, DATA_DIMS, dtype=DTYPE)

    with mp.Pool(2) as p:
        _ = p.starmap(
            set_slot,
            [
                (cache, write_idx_1, write_value_1),
                (cache, write_idx_2, write_value_2),
            ],
        )

    with mp.Pool(2) as p:
        read_value_1, read_value_2 = p.starmap(
            get_slot, [(cache, write_idx_1), (cache, write_idx_2)]
        )

    assert torch.all(read_value_1 == write_value_1)
    assert torch.all(read_value_2 == write_value_2)
