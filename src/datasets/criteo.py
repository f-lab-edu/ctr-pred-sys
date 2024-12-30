import torch
from torch.utils.data import Dataset, DataLoader


class CriteoDataset(Dataset):
    """
    A PyTorch Dataset for the Criteo dataset.
    """

    def __init__(self, numeric_data, categorical_data, labels):
        """
        Initialize the CriteoDataset.

        Parameters:
            numeric_data (pd.DataFrame): A DataFrame containing numeric features.
            categorical_data (pd.DataFrame): A DataFrame containing categorical features.
            labels (pd.Series): A Series containing the target labels.
        """
        self.numeric_data = torch.tensor(numeric_data.values, dtype=torch.float32)
        self.categorical_data = torch.tensor(categorical_data.values, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing numeric features, categorical features, and the label.
        """
        return self.numeric_data[idx], self.categorical_data[idx], self.labels[idx]
