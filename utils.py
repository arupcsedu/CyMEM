"""
# utils.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 22/11/2025
"""

# utils.py
import os
from typing import Tuple


def _get_int_env(*names: str, default: int) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return default


def get_hpc_shard() -> Tuple[int, int]:
    """
    Return (rank, world_size) robustly across:
      - Slurm (SLURM_PROCID / SLURM_NTASKS / SLURM_STEP_NUM_TASKS)
      - PMI (PMI_RANK / PMI_SIZE, PMIX_RANK / PMIX_SIZE)
      - OpenMPI (OMPI_COMM_WORLD_RANK / OMPI_COMM_WORLD_SIZE)
      - Generic env (RANK / WORLD_SIZE)
    """
    rank = _get_int_env(
        "SLURM_PROCID",
        "PMI_RANK",
        "PMIX_RANK",
        "OMPI_COMM_WORLD_RANK",
        "RANK",
        default=0,
    )

    world_size = _get_int_env(
        "SLURM_STEP_NUM_TASKS",   # often more reliable inside srun step
        "SLURM_NTASKS",
        "PMI_SIZE",
        "PMIX_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "WORLD_SIZE",
        default=1,
    )

    # Safety
    if world_size <= 0:
        world_size = 1
    if rank < 0:
        rank = 0
    if rank >= world_size:
        # still run, but avoid crashing sharding
        rank = rank % world_size

    return rank, world_size
