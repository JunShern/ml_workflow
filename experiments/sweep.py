import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict

import wandb
from wandb import env

GROUP_NAME = "experiment_1"
SWEEP_CONFIG: Dict[str, Any] = {
    "name" : GROUP_NAME,
    "program": "my_project.model.train",
    "command": ["${env}", "${interpreter}", "-m", "${program}", "${args}"],

    # Param sweep
    "method" : "grid",
    "parameters" : {
        "group": {
            "values": [GROUP_NAME]
        },
        "tags": {
            "values": [["demo", "look-multiple-tags", "in-a-single-run"]] # Pass list as a single parameter
        },
        "random_seed": {
            "values": [0, 1, 2]
        },
        "train.batch_size": {
            "values": [4, 8, 16]
        },
        "train.learning_rate": {
            "values": [1e-5]
        },
    },
}


@dataclass
class SlurmConfig:
    """
    Params for SLURM compute
    """
    job_name: str = GROUP_NAME # Name of job
    gpu_model: str = "rtx8000" # GPU model
    num_cpus: int = 4 # Number of CPUs per job
    num_gpus: int = 1 # Number of GPUs per job
    maxtime_hours: int = 48 # Maximum runtime of job in hours
    memory_gb: int = 32 # RAM memory in GB


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", "-p", type=str, default=None, help="Launch jobs for an existing wandb sweep")
    parser.add_argument("--num_jobs", "-n", type=int, default=None, help="Number of jobs to launch")
    parser.add_argument('--no_slurm', action='store_true', help="Run on current machine, without slurm")
    
    args = parser.parse_args()

    # Initialize the sweep on the wandb server
    sweep_path = args.sweep_path
    if sweep_path is None:
        sweep_id = wandb.sweep(SWEEP_CONFIG)

        wandb_entity, wandb_project = env.get_entity(), env.get_project()
        sweep_path = f"{wandb_entity}/{wandb_project}/{sweep_id}"
    assert wandb.Api().sweep(sweep_path) # Confirm that sweep_path is valid
    print(f"Using sweep path: {sweep_path}\n")

    # Check how many jobs are needed for this sweep
    if args.num_jobs:
        num_jobs = args.num_jobs
    else:
        if SWEEP_CONFIG['method'] == 'grid':
            # Calculate num_jobs based on sweep params (product of config values)
            num_jobs = 1
            for param, val in SWEEP_CONFIG['parameters'].items():
                if 'values' in val:
                    num_jobs *= max(len(val['values']), 1)
        else:
            print("Unable to infer number of jobs to launch from sweep config. Please specify `--num_jobs`.")
            sys.exit()
    print(f"Launching {num_jobs} jobs for {sweep_path}...\n")

    if args.no_slurm:
        # Run the jobs locally, in series
        cmd = ["wandb", "agent", sweep_path]
    else:
        # Queue the jobs as an sbatch array on SLURM
        slurm_config = SlurmConfig()
        cmd = [
            f"sbatch",
            f"agent.sbatch",
            f"{sweep_path}",
            f"--array=1-{num_jobs}",
            f"--job-name={slurm_config.job_name}",
            f"--cpus-per-task={slurm_config.num_cpus}",
            f"--gres=gpu:{slurm_config.gpu_model}:{slurm_config.num_gpus}",
            f"--mem={slurm_config.memory_gb}GB",
            f"--time={slurm_config.maxtime_hours}:00:00",
        ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("Done.")
