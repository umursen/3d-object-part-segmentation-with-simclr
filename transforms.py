import numpy as np

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')


class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR train dataset
    """

    def __init__(
        self
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # TODO: Add transforms here
        data_transforms = [
        ]

        data_transforms = transforms.Compose(data_transforms)
        self.train_transform = transforms.Compose([data_transforms, transforms.ToTensor()])

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLREvalDataTransform(object):
    """
    Transforms for SimCLR eval dataset
    """

    def __init__(self):
        super().__init__()
        # TODO: Add resize transform if necessary. Otherwise just use cast to a tensor.
        data_transforms = [
        ]

        data_transforms = transforms.Compose(data_transforms)
        self.eval_transform = transforms.Compose([data_transforms, transforms.ToTensor()])

    def __call__(self, sample):
        transform = self.eval_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj
