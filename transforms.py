from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from augmentations.augmentations import RandomCuboid

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


# Self-supervised

class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR
    """

    def __init__(self, data_transforms) -> None:
        augmentations = transforms.Compose([
            RandomCuboid(p=1),
            *data_transforms
        ])
        self.data_transforms = augmentations

        # TODO: Parameterize the crop size
        self.online_transform = transforms.Compose([
            RandomCuboid(p=1),
        ])

    def __call__(self, sample):
        transform = self.data_transforms

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """
    Transforms for SimCLR
    """

    def __init__(self, data_transforms) -> None:
        super().__init__(data_transforms)

    def __call__(self, sample):
        transform = self.data_transforms

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, sample


# Fine-tuning

class FineTuningTrainDataTransform(object):
    """
    Transforms for SimCLR
    """

    def __init__(self, data_transforms) -> None:
        self.data_transforms = transforms.Compose([
            *data_transforms
        ])

    def __call__(self, sample):
        transform = self.data_transforms
        xi = transform(sample)
        return xi

class FineTuningEvalDataTransform(object):
    """
    Transforms for SimCLR
    """

    def __call__(self, sample):
        return sample

class FineTuningTestDataTransform(object):
    """
    Test transforms for SimCLR fine tuning module.
    """

    def __init__(self, data_transforms) -> None:
        self.data_transforms = transforms.Compose([
            *data_transforms
        ])

    def __call__(self, sample):
        transform = self.data_transforms
        xi = transform(sample)
        return xi