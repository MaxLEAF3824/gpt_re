import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

REMOTE_ROOT_URL= "https://rome.baulab.info"
REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/known_1000.json"


class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, size=None, *args, **kwargs,):
        data_dir = Path(data_dir)
        known_loc = data_dir / "known_1000.json"
        if not known_loc.exists():
            print(f"{known_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, known_loc)

        with open(known_loc, "r") as f:
            self.data = json.load(f)
        if size:
            self.data = self.data[:size]
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["prompt"]
