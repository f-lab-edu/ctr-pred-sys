import torch
from torch.utils.data import Dataset, DataLoader


# PyTorch Dataset
class CriteoDataset(Dataset):
    def __init__(self, numeric_data, categorical_data, labels):
        self.numeric_data = torch.tensor(numeric_data.values, dtype=torch.float32)
        self.categorical_data = torch.tensor(categorical_data.values, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.numeric_data[idx], self.categorical_data[idx], self.labels[idx]
