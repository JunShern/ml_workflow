from dataclasses import asdict

import pyrallis
import numpy as np
import wandb

import utils
from config import ExperimentConfig


def train(model: str, epochs: int, learning_rate: float, batch_size: int):
    """
    Train model
    """

    print(f'Training {model} for {epochs} epochs...')
    for epoch in range(1, epochs + 1):

        # Create some mock metrics
        np.random.rand(batch_size) # pump the RNG to get some variability from hyperparams
        train_acc = 1 - (2 ** -(epoch)) - (np.random.rand() / epoch)
        train_loss = (2 ** -(epoch)) + (np.random.rand() / epoch)

        # Log
        wandb.log({
            'epoch': epoch,
            'accuracy': train_acc,
            'loss': train_loss,
        })


@pyrallis.wrap()
def main(cfg: ExperimentConfig):
    
    # Initialize wandb Run; used for logging and tracking experiment configs
    wandb.init(
        tags=cfg.tags,
        config=asdict(cfg),
    )

    # Set random seed for reproducibility
    utils.random_seed(cfg.random_seed)

    # Begin training
    train(cfg.train.model, cfg.train.epochs, cfg.train.learning_rate, cfg.train.batch_size)


if __name__ == '__main__':
    main() # type: ignore
