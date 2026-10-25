"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.

Classification heads. Despite the names (kept from LR-ASD so checkpoint keys
match), these modules only produce logits. The losses live in
:class:`pyasd.models.criterion.Criterion`.
"""

import torch.nn as nn


class lossAV(nn.Module):
	"""Audio-visual head: a linear layer mapping embeddings to 2 logits (not speaking / speaking).

	Args:
		dim (int): Input embedding dimension.
	"""

	def __init__(self, dim):
		super(lossAV, self).__init__()
		self.FC        = nn.Linear(dim, 2)

	def forward(self, x):
		"""Compute the logits.

		Args:
			x (torch.Tensor): Embeddings of shape (B, T, dim).

		Returns:
			torch.Tensor: Logits of shape (B, T, 2).
		"""
		return self.FC(x)


class lossV(nn.Module):
	"""Visual-only auxiliary head: a linear layer mapping embeddings to 2 logits.

	Args:
		dim (int): Input embedding dimension.
	"""

	def __init__(self, dim):
		super(lossV, self).__init__()
		self.FC        = nn.Linear(dim, 2)

	def forward(self, x):
		"""Compute the logits.

		Args:
			x (torch.Tensor): Embeddings of shape (B, T, dim).

		Returns:
			torch.Tensor: Logits of shape (B, T, 2).
		"""
		return self.FC(x)
