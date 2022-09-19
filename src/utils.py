import random
import numpy as np
# import torch

def random_seed(seed: int) -> None:
    """
    Manually set random seed for various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
