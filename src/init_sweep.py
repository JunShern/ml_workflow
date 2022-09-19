from typing import Any, Dict

import wandb


WANDB_ENTITY = "junshern"
WANDB_PROJECT = "ml_workflow"

SWEEP_CONFIG: Dict[str, Any] = {
    "name" : "my-sweep",
    "program": "train.py",

    # Param sweep
    "method" : "grid",
    "parameters" : {
        "tags": {
            "value": ["demo", "multiple-tags"] # Pass the list as a single parameter
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


sweep_id = wandb.sweep(SWEEP_CONFIG, entity=WANDB_ENTITY, project=WANDB_PROJECT)
print(f"Created sweep with path: {WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
