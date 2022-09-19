from pathlib import Path
from typing import List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """
    Params for dataset
    """
    base_dir: Path = Path(".") # Base path of repo
    data_dir: Path =  base_dir / "dataset" # Path to store datasets
    raw_data_dir: Path =  data_dir / "raw" # Subdir within data_dir to store raw data
    processed_data_dir: Path =  data_dir / "processed" # Subdir within data_dir to store processed data

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
class ExperimentConfig:
    """
    Experiment config, can store other configs hierarchically
    """
    tags: List[str] = field(default_factory=list) # tags to apply to the project
    random_seed: int = 0 # Seed for random number generators
    train: TrainConfig = field(default_factory=TrainConfig)
