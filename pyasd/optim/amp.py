"""
Re-export of the torch automatic mixed precision gradient scaler.

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.amp as amp


__all__ = ['GradScaler']

GradScaler = amp.GradScaler
