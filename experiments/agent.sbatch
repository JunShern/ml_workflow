#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
## Place the stdout and stderr into a file called `<job_id>_<arr_idx>.out``
#SBATCH --output=~/slurm_logs/%A_%a.out

## Activate environment
source ~/.bashrc
conda activate workflow

## Print some environment info for posterity
echo "SLURM_JOBID" $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID" $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID" $SLURM_ARRAY_TASK_ID
myquota
nvidia-smi
which python

cd /PATH/TO/ml_workflow # Go to the root of the repo
pwd

# Setup wandb
source env.sh
wandb login

## Run experiment
sweep_path=$1
wandb agent --count 1 $sweep_path # Each node takes one command from the sweep