from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Params for dataset
    """
    base_dir: Path = Path(".").absolute() # Base path of repo
    data_dir: Path =  base_dir / "data" # Path to store datasets
    raw_data_dir: Path =  data_dir / "raw" # Subdir within data_dir to store raw data
    processed_data_dir: Path =  data_dir / "processed" # Subdir within data_dir to store processed data
