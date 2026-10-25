"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.
"""

import torch.nn as nn
import torch.nn.functional as F

from pyasd.models.lightasd.loss import lossAV, lossV
from pyasd.models.lightasd.encoder import visual_encoder, audio_encoder
from pyasd.models.lightasd.classifier import BGRU


class LightASD(nn.Module):
    """Light-ASD active speaker detection model.

    Lightweight CNN encoders give 128-dim audio and visual embeddings per video
    frame. They are summed and passed through a bidirectional GRU and a linear
    classifier. A visual-only auxiliary head is used during training.

    Adapted from Light-ASD (https://github.com/Junhua-Liao/Light-ASD).
    """
    def __init__(self):
        """Build the encoders, the GRU backend and the classification heads.

        Takes no arguments.
        """
        super(LightASD, self).__init__()
        
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audio_encoder()
        self.GRU = BGRU(128)
        self.av_head = lossAV()
        self.v_head = lossV()

        self.postprocess = 'softmax'

    def forward_visual_frontend(self, x):
        """Encode a face-crop video into per-frame visual embeddings.

        Frames are normalised with a fixed mean/std (0.4161, 0.1688) and reshaped to
        (B, 1, T, 112, 112) for the 3D-conv visual encoder.

        Args:
            x (torch.Tensor): Video of shape (B, T, 1, 112, 112), grayscale in [0, 1].

        Returns:
            torch.Tensor: Visual embeddings of shape (B, T, 128).
        """
        x = x[:, :, 0, :, :]
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        """Encode MFCC features into per-video-frame audio embeddings.

        Args:
            x (torch.Tensor): MFCC features of shape (B, 13, 4*T).

        Returns:
            torch.Tensor: Audio embeddings of shape (B, T, 128).
        """
        x = x.unsqueeze(1)#.transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):  
        """Fuse audio and visual embeddings by summation and a bidirectional GRU.

        Args:
            x1 (torch.Tensor): Audio embeddings of shape (B, T, 128).
            x2 (torch.Tensor): Visual embeddings of shape (B, T, 128).

        Returns:
            torch.Tensor: Fused features of shape (B, T, 128).
        """
        x = x1 + x2 
        x = self.GRU(x)  
        return x    

    def forward_visual_backend(self,x):
        """Identity backend for the visual auxiliary branch.

        Args:
            x (torch.Tensor): Visual embeddings of shape (B, T, 128).

        Returns:
            torch.Tensor: ``x`` unchanged.
        """
        return x

    def forward(self, visualFeature, audioFeature):
        """Predict per-frame speaking activity.

        Args:
            visualFeature (torch.Tensor): Face-crop video of shape (B, T, 1, 112, 112),
                grayscale in [0, 1].
            audioFeature (torch.Tensor): MFCC features of shape (B, 13, 4*T).

        Returns:
            dict: ``'pred_logits'``: audio-visual logits of shape (B, T, 2). In training
            mode also ``'aux_video_outputs'``: a list with one dict holding the
            visual-only ``'pred_logits'`` of shape (B, T, 2). In eval mode also
            ``'pred_scores'``: softmax of ``'pred_logits'``, shape (B, T, 2).
        """
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        logits = self.av_head(outsAV)  
        
        outputs = {'pred_logits': logits}

        if self.training:
            outsV = self.forward_visual_backend(visualEmbed)
            outputs['aux_video_outputs'] = [{'pred_logits': self.v_head(outsV)}]
        else:
            scores = F.softmax(logits, dim=-1)
            outputs['pred_scores'] = scores

        return outputs  
