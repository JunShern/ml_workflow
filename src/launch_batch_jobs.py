import argparse
import sys

import wandb


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", "-p", type=str, required=True, help="wandb sweep path for jobs to run on")
    parser.add_argument("--num_jobs", "-n", type=int, default=None, help="number of jobs to launch")
    
    args = parser.parse_args()

    if args.num_jobs:
        num_jobs = args.num_jobs
    else:
        # Try to calculate from sweep params
        api = wandb.Api()
        sweep = api.sweep(path=args.sweep_path)
        if sweep.config['method'] == 'grid':
            # Number of jobs to launch is the product of config values
            num_jobs = 1
            for param, val in sweep.config['parameters'].items():
                if 'values' in val:
                    num_jobs *= len(val['values'])
        else:
            print("Unable to infer number of jobs to launch from sweep config. Please specify `--num_jobs`.")
            sys.exit()
    
    print(f"Launching {num_jobs} batch jobs...")


    # TODO: Launch sbatch array job with the correct number of sweeps
    cmd = f"sbatch agent.sbatch {args.sweep_path}"
    print(cmd)
