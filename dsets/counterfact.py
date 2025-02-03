import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset
from experiments.utils import predict_token

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/counterfact.json"


class CounterFactDataset(Dataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / "counterfact.json"
        if not cf_loc.exists():
            print(f"{cf_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, cf_loc)

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        preds = predict_token(
            mt,
            [self.data[i]['prompt'] for i in range(len(self.data))],
            topk=1
        )

        self.data = [
            self.data[i] for i in range(len(self.data)) 
            if preds[i][0].strip() == self.data[i]['attribute']
        ]

        for d in self.data:
            d['id'] = d.pop('case_id')
            d['subject'] = d["requested_rewrite"]['subject']
            d['prompt'] = d["requested_rewrite"]["prompt"].replace("{}", self.data[item]['subject'])
            d['target'] = d["requested_rewrite"]["target_true"]["str"]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
