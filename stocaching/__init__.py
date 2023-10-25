"""Stocaching, a tiny library for stochastic dataset caching in PyTorch."""

from stocaching.shared_cache import SharedCache
from stocaching.utils import get_shm_size

__all__ = ["SharedCache", "get_shm_size"]
