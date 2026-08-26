import json
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, path, max_len):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "prompt": item.get("prompt", item.get("instruction", "")),
            "reference": item.get("completion", item.get("output", ""))
        }

    def collate(self, batch):
        return {
            "prompts": [b["prompt"] for b in batch],
            "references": [b["reference"] for b in batch]
        }
