import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *



class LongRADataset(Dataset):
    def __init__(self, *args, **kwargs):
        longra_loc = Path("data/longra.json")
        if not longra_loc.exists():
            print(f"{longra_loc} does not exist. Please download it first.")

        with open(longra_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
