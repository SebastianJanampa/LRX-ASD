import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class attentionLayer(nn.Module):
    """Transformer-style attention block used for cross- and self-attention.

    Multi-head attention (queries from ``tar``, keys/values from ``src``) followed by
    a position-wise feed-forward network, each with dropout, a residual connection
    and post-LayerNorm.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        """Initialise the attention block.

        Args:
            d_model (int): Feature dimension of the inputs and outputs.
            nhead (int): Number of attention heads.
            dropout (float): Dropout probability for the attention weights, the
                feed-forward network and the residual branches. Defaults to 0.1.
        """
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        """Attend from ``tar`` (queries) to ``src`` (keys/values) and apply the FFN.

        Args:
            src (torch.Tensor): Key/value sequence of shape (B, T, d_model). Also used
                as the residual input.
            tar (torch.Tensor): Query sequence of shape (B, T, d_model).

        Returns:
            torch.Tensor: Updated sequence of shape (B, T, d_model).
        """
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src
