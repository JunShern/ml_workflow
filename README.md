# ml_workflow

Skeleton/template for ML research codebase, accompanies this blog post on my [ML Experiment Workflow](https://junshern.notion.site/ML-Experiment-Workflow-f01d38f80ba745468aef23284f3f6e37).

Repo overview (adapted from [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science)):
```
├── README.md          <- The README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── env.sh             <- Script to define and set environment variables
│
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── my_project         <- Source code, replace with your project name
│   ├── __init__.py    <- Makes my_project a Python module
│   │
│   ├── common         <- Modules shared by different parts of the project
│   │   └── utils.py
│   │
│   ├── data           <- Scripts to download or generate data
│   │   ├── config.py
│   │   └── make_dataset.py
│   │
│   └── model          <- Scripts to train and test models
│       ├── config.py
│       ├── test.py
│       └── train.py
│
└── experiments        <- Configs and scripts for running experiments
    ├── agent.sbatch   <- sbatch file to run jobs on SLURM
    └── sweep.py       <- Run hyperparameter sweeps
```

Setup:
```bash
# Get dependencies
conda create -n workflow python=3.9
conda activate workflow
pip install -r requirements.txt

# Fill in the environment variables in `env.sh` then apply them
source env.sh
```

Example usage:
```bash
# Train a single model
python -m my_project.model.train

# Run hyperparameter sweep
python experiments/sweep.py
```
