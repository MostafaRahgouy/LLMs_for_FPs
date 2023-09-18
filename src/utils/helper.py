import numpy as np
import random
import torch
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
