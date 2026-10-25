"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch
import torch.nn as nn

import math
from copy import deepcopy

from pyasd.misc import dist_utils

__all__ = ['ModelEMA']



class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay: float=0.9999, warmups: int=1000, start: int=0):
        """Create a frozen, eval-mode copy of the model to hold the EMA weights.

        Args:
            model: Model to track. DP/DDP wrappers are removed before copying.
            decay: Target EMA decay factor.
            warmups: Time constant (in updates) of the exponential decay ramp-up.
                If 0, ``decay`` is used from the first update.
            start: Number of initial ``update`` calls to skip before averaging.
        """
        super().__init__()

        self.module = deepcopy(dist_utils.de_parallel(model)).eval()

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        if warmups == 0:
            self.decay_fn = lambda x: decay
        else:
            self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)


    def update(self, model: nn.Module):
        """Blend the current model weights into the EMA weights.

        Only floating-point entries of the state dict are averaged. The first
        ``start`` calls are skipped.

        Args:
            model: Model being trained (may be wrapped in DP/DDP).
        """
        if self.before_start < self.start:
            self.before_start += 1
            return
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = dist_utils.de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        """Move or cast the EMA module, as in ``nn.Module.to``.

        Args:
            *args: Positional arguments forwarded to ``nn.Module.to``.
            **kwargs: Keyword arguments forwarded to ``nn.Module.to``.

        Returns:
            ModelEMA: ``self``.
        """
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self, ):
        """Return the EMA state for checkpointing.

        Returns:
            dict: ``{'module': <EMA module state_dict>, 'updates': <update count>}``.
        """
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        """Restore the EMA state produced by ``state_dict``.

        Args:
            state: Dict with a ``'module'`` state dict and an optional ``'updates'`` count.
            strict: Passed to ``nn.Module.load_state_dict``.

        Returns:
            tuple: Two empty lists, mimicking ``(missing_keys, unexpected_keys)``.
        """
        self.module.load_state_dict(state['module'], strict=strict)
        if 'updates' in state:
            self.updates = state['updates']
        return [], []

    def forwad(self, ):
        """Raise an error; the EMA wrapper is not meant to be called directly.

        Note: the name is a typo for ``forward`` inherited from the upstream code.
        Nothing calls it; use ``self.module`` for inference.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError('ema...')

    def extra_repr(self) -> str:
        """Return a short summary of the EMA settings.

        Returns:
            str: The decay and warmup values.
        """
        return f'decay={self.decay}, warmups={self.warmups}'



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu", use_buffers=True):
        """Wrap the model in an ``AveragedModel`` that uses an EMA averaging function.

        Args:
            model: Model whose parameters are averaged.
            decay: Target EMA decay; ramped up with a fixed 2000-step time constant.
            device: Device on which the averaged copy is kept.
            use_buffers: Whether to average buffers as well as parameters.
        """

        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            """Compute one EMA step for a parameter.

            Args:
                avg_model_param: Current averaged parameter.
                model_param: Current model parameter.
                num_averaged: Number of averaging steps taken so far.

            Returns:
                torch.Tensor: The updated averaged parameter.
            """
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)
