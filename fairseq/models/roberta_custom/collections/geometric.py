import torchvision.transforms.functional as TF
from torchvision import transforms
import random
from typing import List

from .base import FeatureAugmentationBase

# geometric transformations includes:
# Rotate
# flip horizontally
# flip vertically


class Rotate(FeatureAugmentationBase):
    def __init__(self, options: List=[], prob: float=1.0) -> None:
        super().__init__(options, prob)

    def my_transform(self, x):
        angle = random.choice(self.options)
        if self.prob < random.random():
            return TF.rotate(x, angle)
        return x


class RandomHorizontalFlip(FeatureAugmentationBase):
    def __init__(self, options: List = [], prob: float = 1.0) -> None:
        super().__init__(options, prob)

    def my_transform(self, x):
        if self.prob < random.random():
            return TF.hflip(x)
        return x


class RandomVerticalFlip(FeatureAugmentationBase):
    def __init__(self, options: List = [], prob: float = 1.0) -> None:
        super().__init__(options, prob)

    def my_transform(self, x):
        if self.prob < random.random():
            return TF.vflip(x)
        return x