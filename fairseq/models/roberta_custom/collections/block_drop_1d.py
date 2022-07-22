from numpy import block
import torch
import random 
import torch.nn.functional as F

from torch import nn



class DropBlock1D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, scheduler_params=None):
        super(DropBlock1D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.scheduler_params = scheduler_params
        self._step = 0
        if self.scheduler_params is not None:
            self.start_value = self.scheduler_params.get('start_value', 0.0)
            self.end_value = self.scheduler_params.get('end_value', 0.25)
            self.num_steps = self.scheduler_params.get('num_steps', 10000)
            self.drop_prob = self.start_value

    def scheduler_step(self):
        if self.scheduler_params is not None:
            self.drop_prob = self.start_value + (self.end_value - self.start_value) * (self._step / self.num_steps)
            self.drop_prob = min(self.drop_prob, self.end_value)
            if (self._step % (1000) == 0):
                print(self._step, self.drop_prob, self.num_steps)

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 3, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or (self.drop_prob == 0. and self.scheduler_params is None):
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            # step
            self._step += 1
            self.scheduler_step()
            #self._step += out.shape[1]
            return out.half()

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(input=mask[:, None, :],
                                  kernel_size=self.block_size,
                                  stride=1,
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size)



class DropBlockChannel1D(nn.Module):
    def __init__(self, drop_prob, block_size, scheduler_params=None):
        super(DropBlockChannel1D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.scheduler_params = scheduler_params
        self._step = 0
        if self.scheduler_params is not None:
            self.start_value = self.scheduler_params.get('start_value', 0.0)
            self.end_value = self.scheduler_params.get('end_value', 0.25)
            self.num_steps = self.scheduler_params.get('num_steps', 10000)
            self.drop_prob = self.start_value

    def scheduler_step(self):
        if self.scheduler_params is not None:
            self.drop_prob = self.start_value + (self.end_value - self.start_value) * (self._step / self.num_steps)
            self.drop_prob = min(self.drop_prob, self.end_value)
            if (self._step % (1000) == 0):
                print(self._step, self.drop_prob, self.num_steps)

    def forward(self, x):

        assert x.dim() == 3, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or (self.drop_prob == 0. and self.scheduler_params is None):
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(*x.shape) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            # step
            self.scheduler_step()
            self._step += 1

            return out.half()

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size)


class AdaptiveDropBlockChannel1D(nn.Module):
    def __init__(self, drop_prob, block_size, scheduler_params=None):
        super(AdaptiveDropBlockChannel1D, self).__init__()

        self.drop_prob = drop_prob
        # self.threshold = threshold
        self.block_size = block_size
        self.scheduler_params = scheduler_params
        self._step = 0
        if self.scheduler_params is not None:
            self.start_value = self.scheduler_params.get('start_value', 0.0)
            self.end_value = self.scheduler_params.get('end_value', 0.25)
            self.num_steps = self.scheduler_params.get('num_steps', 10000)
            self.drop_prob = self.start_value

    def scheduler_step(self):
        if self.scheduler_params is not None:
            self.drop_prob = self.start_value + (self.end_value - self.start_value) * (self._step / self.num_steps)
            self.drop_prob = min(self.drop_prob, self.end_value)

    def forward(self, x):

        assert x.dim() == 3, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or (self.drop_prob == 0. and self.scheduler_params is None):
            return x
        else:
            # if self.drop_prob < random.random():
            #     return x
            # get gamma value
            # gamma = self._compute_gamma(x)
            # biased mask generation
            mask = self._thresholding(x)
            # import pdb; pdb.set_trace()
            # mask = torch.zeros(*x.shape)
            # mask[indices] = 1
            # mask = mask.float()

            # mask = (torch.rand(*x.shape) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            '''
            if self.drop_prob != 0.0:
                print(self.drop_prob, block_mask.sum() / block_mask.numel())
            '''

            # step
            self._step += 1
            self.scheduler_step()
            return out.half()

    def _thresholding(self, x):
        x = torch.abs(x)
        thresholds = F.avg_pool2d(
            input=x,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            thresholds = thresholds[:, :, :-1]
        gamma = self._compute_gamma()
        tops, top_indices = torch.topk(
            thresholds.flatten(), int(x.numel() * gamma))
        return (thresholds > tops[-1]).float()

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask,
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self):
        return self.drop_prob / (self.block_size)



class ReverseAdaptiveDropBlockChannel1D(AdaptiveDropBlockChannel1D):
    def __init__(self, drop_prob, block_size, scheduler_params=None):
        super(ReverseAdaptiveDropBlockChannel1D, self).__init__(
            drop_prob=drop_prob, 
            block_size=block_size, 
            scheduler_params=scheduler_params)
        self.pool = torch.nn.AvgPool1d(kernel_size=block_size, stride=1, padding=block_size//2)
        # self.upsample = torch.nn.ConvTranspose2d(channel_size, channel_size, block_size, bias=False)
        # self.upsample.weight.data.fill_(1.0)

    def forward(self, x):
        assert x.dim() == 3, \
            "Expected input with 3 dimensions (bsize, channels, height, width)"

        if not self.training or (self.drop_prob == 0. and self.scheduler_params is None):
            return x
        else:
            # if self.drop_prob < random.random():
            #     return x
            # biased mask generation
            mask = self._thresholding(x)
            # import pdb; pdb.set_trace()
            # mask = torch.zeros(*x.shape)
            # mask[indices] = 1
            # mask = mask.float()

            # mask = (torch.rand(*x.shape) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            # block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * mask
            out = out * mask.numel() / mask.sum()

            '''
            if self.drop_prob != 0.0:
                print(self.drop_prob, block_mask.sum() / block_mask.numel())
            '''

            # step
            self._step += 1
            self.scheduler_step()
            return out.half()

    def _thresholding(self, x):
        #print(self.drop_prob, 'inside thresholding')

        # pooled = self.pool(x)
        thresholds = torch.abs(x)

        gamma = self._compute_gamma()
        mean_threshold = torch.mean(thresholds)

        # add noise
        randomised_thresholds = thresholds + 0.1 * torch.rand(*thresholds.shape).to(thresholds.device) * mean_threshold 
        topk = int(randomised_thresholds.numel() * gamma)
        randomised_thresholds_flat = randomised_thresholds.flatten()

        if topk <= 0:
            bar = 0
        else:
            tops, top_indices = torch.topk(
                randomised_thresholds_flat, topk,
                largest=False)
            bar = randomised_thresholds_flat[top_indices][-1]
        mask = (randomised_thresholds < bar).float()
        res = self.pool(mask)

        # print('gamma', gamma, 'mask', mask.sum()/mask.numel())
        # print((res>0).sum()/res.numel())
        if self.block_size % 2 == 0:
            res = res[:, :, :-1]
        return (res == 0).float()
        # print(topk, gamma, bar)
        # # # print(tops, thresholds)
        # return (x < bar).float()


