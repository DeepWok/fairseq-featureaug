import torch
import random
from typing import List, Tuple, Optional, Dict


class FeatureAugmentationBase:

    def __init__(
            self, 
            options: List=[], 
            prob: float=1.0) -> None:
        self.options = options
        self.prob = prob

    def __call__(self, x):
        fn = self.my_transform
        shapes = x.shape
        y = [fn(x[:, i, :, :]) for i in range(shapes[1])]
        y = torch.stack(y, axis=1)

        '''channel wise, but it is really  slow
        reshaped_x = x.reshape(shapes[0]*shapes[1], shapes[2], shapes[3])
        y = [fn(reshaped_x[i, :, :].unsqueeze(0)) for i in range(shapes[0]*shapes[1])]
        y = torch.stack(y)
        y = y.reshape(shapes[0], shapes[1], shapes[2], shapes[3])
        '''
        return y
        # mypick = random.choice(self.options)
        # return TF.rotate(x, angle)
    
    def my_transform(self, x):
        raise NotImplementedError()
