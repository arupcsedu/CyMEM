"""
# utils.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 22/11/2025
"""


import os
from typing import Tuple


def get_hpc_shard() -> Tuple[int, int]:
    """
    Determine (rank, world_size) from SLURM or generic distributed env vars.

    Fallback to (0, 1) for single-process runs.
    """
    # SLURM
    rank = os.environ.get("SLURM_PROCID")
    world_size = os.environ.get("SLURM_NTASKS")

    # Generic (e.g., torch.distributed or other launchers)
    if rank is None:
        rank = os.environ.get("RANK", "0")
    if world_size is None:
        world_size = os.environ.get("WORLD_SIZE", "1")

    try:
        r = int(rank)
    except Exception:
        r = 0
    try:
        ws = int(world_size)
    except Exception:
        ws = 1

    return r, ws
