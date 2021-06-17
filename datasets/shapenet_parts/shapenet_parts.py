import torch

class ShapeNetParts(torch.utils.data.Dataset):
    def __init__(self, split):
        assert split in ['train', 'val', 'overfit']
        pass
    def __getitem__(self, index):
        pass

    def __len__(self):
        return None
        
    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        pass
