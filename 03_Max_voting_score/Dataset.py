import numpy as np

from torch.utils.data import Dataset

from skimage import exposure


class SlicesDataset(Dataset):
    def __init__(self, x, y, transform=False):
        """
        Args:
            x (np.array): array of ROIs slices
            y (np.array): array of labels
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.transform:
            x = exposure.equalize_adapthist(self.x[index])
            y = self.y[index]
        else:
            x = self.x[index]
            y = self.y[index]

        # Repeat each slice three times on the z-axis to have a tensor of shape [3, 64, 64]
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, 3, axis=0)
        
        return x, y
