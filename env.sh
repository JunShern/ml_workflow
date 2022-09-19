### PROJECT VARIABLES
export REPO_PATH="/ABS/PATH/TO/ml_workflow"

### WANDB VARIABLES
# https://docs.wandb.ai/guides/track/advanced/environment-variables

# This is secret and shouldn't be checked into version control
export WANDB_API_KEY=$YOUR_API_KEY

export WANDB_ENTITY="junshern"
export WANDB_PROJECT="ml_workflow"

# If you want to turn off wandb
# export WANDB_MODE="disabled" # offline/disabled/online

# WANDB_DIR: Set this to an absolute path to store all generated files here instead of
# the wandb directory relative to your training script. be sure this directory exists 
# and the user your process runs as can write to it.
