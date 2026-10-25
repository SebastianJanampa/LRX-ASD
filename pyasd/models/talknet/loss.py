"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.
"""

import torch.nn as nn

class lossAV(nn.Module):
	"""Linear audio-visual classification head (256-dim feature -> 2 logits).

	Despite its name, this module computes no loss.

	Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
	"""
	def __init__(self):
		"""Initialise the 256 -> 2 linear layer. Takes no arguments."""
		super(lossAV, self).__init__()
		self.FC        = nn.Linear(256, 2)
		
	def forward(self, x):	
		"""Compute speaking / not-speaking logits.

		Args:
			x (torch.Tensor): Audio-visual features of shape (B, T, 256).

		Returns:
			torch.Tensor: Logits of shape (B, T, 2).
		"""
		return self.FC(x)

class lossV(nn.Module):
	"""Linear single-modality classification head (128-dim feature -> 2 logits).

	Used as the auxiliary visual-only head. Despite its name, this module computes
	no loss.

	Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
	"""
	def __init__(self):
		"""Initialise the 128 -> 2 linear layer. Takes no arguments."""
		super(lossV, self).__init__()
		self.FC        = nn.Linear(128, 2)

	def forward(self, x):	
		"""Compute speaking / not-speaking logits.

		Args:
			x (torch.Tensor): Single-modality features of shape (B, T, 128).

		Returns:
			torch.Tensor: Logits of shape (B, T, 2).
		"""
		return self.FC(x)
