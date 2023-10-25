import os


def get_shm_size() -> int:
    """Get size of /dev/shm. This is often a good number to set as a maximum cache size
    Returns:
        (int) Size of /dev/shm in GiB
    """
    bytes_per_gib = 1024**3
    stats = os.statvfs("/dev/shm")
    shm_bytes = stats.f_bsize * stats.f_blocks
    shm_size = shm_bytes / bytes_per_gib
    return int(shm_size)
