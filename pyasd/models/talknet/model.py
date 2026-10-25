import torch
import torch.nn as nn
import torch.nn.functional as F

from pyasd.models.talknet.loss import lossAV, lossV
from pyasd.models.talknet.audioEncoder import audioEncoder
from pyasd.models.talknet.visualEncoder import visualFrontend, visualTCN, visualConv1D
from pyasd.models.talknet.attentionLayer import attentionLayer

class TalkNet(nn.Module):
    """TalkNet active speaker detection model.

    Audio and visual temporal encoders produce 128-dim embeddings per video frame.
    These are concatenated and passed through an audio-visual self-attention layer
    and a linear classifier. A visual-only auxiliary head is used during training.

    Note: the cross-attention layers (``crossA2V``, ``crossV2A``) and ``a_head`` are
    built but not used by ``forward``.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self):
        """Build the encoders, attention layers and classification heads.

        Takes no arguments.
        """
        super(TalkNet, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)

        self.av_head = lossAV()
        self.v_head = lossV()
        self.a_head = lossV()
        
        self.postprocess = 'softmax'

    def forward_visual_frontend(self, x):
        """Encode a face-crop video into per-frame visual embeddings.

        Frames are normalised with a fixed mean/std (0.4161, 0.1688), passed through
        the 3D-conv + ResNet18 frontend (512-dim per frame), the visual TCN and a 1D
        conv projection to 128 dims.

        Args:
            x (torch.Tensor): Video of shape (B, T, 1, 112, 112), grayscale in [0, 1].

        Returns:
            torch.Tensor: Visual embeddings of shape (B, T, 128).
        """
        x = x[:, :, 0, :, :]
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        """Encode MFCC features into per-video-frame audio embeddings.

        Args:
            x (torch.Tensor): MFCC features of shape (B, 13, 4*T).

        Returns:
            torch.Tensor: Audio embeddings of shape (B, T, 128).
        """
        x = x.unsqueeze(1)       
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        """Apply audio-visual cross-attention. Not used by ``forward``.

        Args:
            x1 (torch.Tensor): Audio embeddings of shape (B, T, 128).
            x2 (torch.Tensor): Visual embeddings of shape (B, T, 128).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``x1`` attended with queries from ``x2``
            and ``x2`` attended with queries from ``x1``, each of shape (B, T, 128).
        """
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2): 
        """Fuse audio and visual embeddings with self-attention.

        Args:
            x1 (torch.Tensor): Audio embeddings of shape (B, T, 128).
            x2 (torch.Tensor): Visual embeddings of shape (B, T, 128).

        Returns:
            torch.Tensor: Fused features of shape (B, T, 256).
        """
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(src = x, tar = x)       
        return x    

    def forward_audio_backend(self,x):
        """Identity backend for the audio branch. Not used by ``forward``.

        Args:
            x (torch.Tensor): Audio embeddings of shape (B, T, 128).

        Returns:
            torch.Tensor: ``x`` unchanged.
        """
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
