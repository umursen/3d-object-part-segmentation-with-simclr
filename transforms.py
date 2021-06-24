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
    Transforms for SimCLR
    """

    def __init__(self, data_transforms) -> None:
        self.data_transforms = transforms.Compose(data_transforms)
        # print(self.data_transforms)

    def __call__(self, sample):
        transform = self.data_transforms

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLREvalDataTransform(object):
    """
    Transforms for SimCLR
    """

    def __init__(self, data_transforms) -> None:
        self.data_transforms = transforms.Compose(data_transforms)
        # print(self.data_transforms)

    def __call__(self, sample):
        transform = self.data_transforms

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj
