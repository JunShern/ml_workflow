from typing import List, Optional
from dataclasses import dataclass, field

from my_project.data.config import DatasetConfig


@dataclass
class TrainConfig:
    """
    Params for model training
    """
    batch_size: int = 4 # Batch size
    epochs: int = 5 # The number of epochs to train
    learning_rate: float = 1e-5 # Learning rate
    model: str = "gpt-35" # The model name


@dataclass
class TestConfig:
    """
    Params for model evaluation
    """
    batch_size: int = 4 # Test batch size
    test_samples: int = 100 # Num samples to use from test set
    

@dataclass
class RunConfig:
    """
    Parent config for a run, can store other configs hierarchically
    """
    group: Optional[str] = None # Name for this experiment group
    tags: List[str] = field(default_factory=list) # Tags to apply to this run
    job_type: str = "train" # Type of job (train|test)
    random_seed: int = 0 # Seed for random number generators
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
