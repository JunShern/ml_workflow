import os
import random
from typing import Optional
import numpy as np
# import torch


def random_seed(seed: int) -> None:
    """
    Manually set random seed for various libraries.
    """

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.device_count() > 0:
    #     torch.cuda.manual_seed_all(seed)


def get_slurm_id() -> Optional[str]:
    """
    Checks for a valid SLURM ID in the environment variables.
    """

    slurm_job_id = None
    if 'SLURM_ARRAY_JOB_ID' in os.environ and 'SLURM_ARRAY_TASK_ID' in os.environ:
        slurm_job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    elif 'SLURM_JOBID' in os.environ:
        slurm_job_id = os.environ['SLURM_JOBID']
    return slurm_job_id