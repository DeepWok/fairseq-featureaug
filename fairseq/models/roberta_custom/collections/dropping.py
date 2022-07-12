import random
import numpy as np
import torchvision.transforms.functional as TF
import torch
import functools
import itertools

from torchvision import transforms
from typing import List, Tuple, Dict

from .base import FeatureAugmentationBase

from multiprocessing import Pool


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def erase(x_w, x_h, block_w_ratio, block_h_ratio):
    x = np.ones((x_w, x_h))
    # get a central point
    bw_l, bw_h = block_w_ratio[0], block_w_ratio[1]
    bh_l, bh_h = block_h_ratio[0], block_h_ratio[1]
    block_w = random.randint(int(x_w*bw_l), int(x_w*bw_h))
    block_h = random.randint(int(x_h*bh_l), int(x_h*bh_h))

    central_w = random.randint(0, x_w)
    central_h = random.randint(0, x_h)

    top_left_pos = (max(0, int(central_w - block_w/2)), max(0, int(central_h - block_h/2)))
    bot_right_pos = (min(x_w, int(central_w + block_w/2)), min(x_h, int(central_h + block_h/2)))
    x[top_left_pos[0]: bot_right_pos[0], top_left_pos[1]: bot_right_pos[1]] = 0
    return x

def erase_list(n, w, h, bw, bh):
    res = [] 
    for i in n:
        res.append(erase(w, h, bw, bh))
    return res


# class RandomErasingChannelWise(FeatureAugmentationBase):
class RandomErasingTest(FeatureAugmentationBase):
    def __init__(
            self,
            options: List = [],
            prob: float = 1.0) -> None:
        super().__init__(options, prob)
        self.block_w_ratios = (0.5, 0.8)
        self.block_h_ratios = (0.5, 0.8)
    
    def parallel_fn(self, x):
        if self.prob < random.random():
            if self.options == []:
                configs = {}
            else:
                configs = random.choice(self.options)
            print(x.shape)
            fn = transforms.RandomErasing(p=1.0, **configs)
            return fn(x)

    def my_transform(self, x):
        if len(x.shape) == 3:
            c, h, w = x.shape
            b = 1
        else:
            b, c, h, w = x.shape
        ms = []
        for i in range(b*c):
            m = erase(w, h, self.block_w_ratios, self.block_h_ratios)
            ms.append(m)
        '''
        mylist = [i for i in range(b*c)] 
        chunked = chunks(mylist, 3)
        pool = Pool(processes=3)
        erase_list_fn = functools.partial(erase_list, w=w, h=h, bw=self.block_w_ratios, bh=self.block_h_ratios)
        ms = pool.map(erase_list_fn, chunked)
        pool.close()
        pool.join()
        ms = list(itertools.chain.from_iterable(ms))
        '''

        ms = np.array(ms)
        if len(x.shape) == 3:
            ms = ms.reshape(c, h, w)
        else:
            ms = ms.reshape(b, c, h, w)
        ms = torch.from_numpy(ms).to(x.device).float()
        # print(ms.sum()/ms.numel())
        # tmp = x
        x = x * ms
        # import pdb; pdb.set_trace()
        return x


class RandomErasing(FeatureAugmentationBase):
    def __init__(
            self,
            options: List=[], 
            prob: float=1.0) -> None:
        super().__init__(options, prob)

    def my_transform(self, x):
        if self.prob < random.random():
            if self.options == []:
                configs = {}
            else:
                configs = random.choice(self.options)
            fn = transforms.RandomErasing(p=1.0, **configs)
            return fn(x)
        return x



class RandomCrop(FeatureAugmentationBase):
    def __init__(self, options: List = [], prob: float = 1.0) -> None:
        super().__init__(options, prob)

    def my_transform(self, x):
        if self.prob < random.random():
            if self.options == []:
                configs = {}
            else:
                configs = random.choice(self.options)
                _, _, size, _ = x.shape
                configs = {'size': int(size * configs['size'])}
            fn = transforms.RandomCrop(**configs)
            return fn(x)
        return x


