# from .geometric import Rotate, RandomHorizontalFlip, RandomVerticalFlip
# from .dropping import RandomCrop, RandomErasing, RandomErasingTest
# from .block_drop import DropBlock2D, DropBlockChannel2D, AdaptiveDropBlockChannel2D, ReverseAdaptiveDropBlockChannel2D
from .block_drop_1d import DropBlock1D, DropBlockChannel1D, ReverseAdaptiveDropBlockChannel1D
# from .config import Configs


def get_transform(name, args):
    if name == 'drop_block':
        return DropBlock1D(**args)
    elif name == 'drop_block_head':
        return DropBlockChannel1D(**args)
    elif name == 'adaptive':
        return ReverseAdaptiveDropBlockChannel1D(**args)
