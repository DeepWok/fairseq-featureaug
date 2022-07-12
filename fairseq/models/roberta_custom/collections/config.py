import torch
import yaml
import torchvision.transforms as transforms
# from collections import OrderedDict
from .dropping import (RandomCrop, RandomErasing, RandomErasingTest)
from .geometric import (Rotate, RandomHorizontalFlip, RandomVerticalFlip)
from .block_drop import (DropBlock2D, DropBlockChannel2D, AdaptiveDropBlockChannel2D, ReverseAdaptiveDropBlockChannel2D)



cls_map = {
    'RandomCrop': RandomCrop,
    'RandomErasing': RandomErasing,
    'RandomErasingTest': RandomErasingTest,
    'Rotate': Rotate,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'DropBlock2D': DropBlock2D,
    'DropBlockChannel2D': DropBlockChannel2D,
    'AdaptiveDropBlockChannel2D': AdaptiveDropBlockChannel2D,
    'ReverseAdaptiveDropBlockChannel2D': ReverseAdaptiveDropBlockChannel2D,
}


class Configs:
    def __init__(self, file_name) -> None:
        self.configs = yaml.safe_load(open(file_name, 'r'))
        self._build_transforms()
    
    def _build_transforms(self) -> None:
        my_transforms = []
        for transform_name, transform_args in self.configs.items():
            if transform_name not in cls_map:
                raise ValueError(f'{transform_name} is not supported')
            cls = cls_map[transform_name]
            t = cls(**transform_args)
            my_transforms.append(t)
        # self.transforms = transforms.Compose(my_transforms)
        # self.transforms = torch.nn.ModuleList(my_transforms)
        self.transforms = my_transforms[0]
    
    def get_transforms(self):
        return self.transforms
