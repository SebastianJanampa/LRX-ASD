"""
Modified from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.

LRX-ASD: lightweight audio-visual active speaker detection with
re-parameterizable encoders, a RoPE transformer detector and an audio-visual ->
visual DINO-style self-distillation loss.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyasd.models.lrxasd.loss import lossAV, lossV
from pyasd.models.lrxasd.dino import DINOHead
from pyasd.models.lrxasd.encoder import visual_encoder, audio_encoder
from pyasd.models.lrxasd.classifier import Fusion2, DetectorAttention


class LRXASDBase(nn.Module):
    """LRX-ASD model.

    Pipeline: visual encoder and audio encoder -> fusion -> transformer
    detector -> audio-visual head. During training, a visual-only head and a
    DINO head (teacher: audio-visual features, student: visual features) add
    auxiliary outputs. :meth:`deploy` removes both and fuses the RepVGG branches.

    Use the ready-made variants :data:`LRXASD`, :data:`LRXASDSmall` and
    :data:`LRXASDTiny`.

    Args:
        dims (list[int]): Channels of the three encoder stages. ``dims[-1]`` is
            the embedding dimension used by the fusion, detector and heads.
    """

    def __init__(self, dims):
        super(LRXASDBase, self).__init__()

        self.visualEncoder  = visual_encoder(dims=dims)
        self.audioEncoder  = audio_encoder(dims=dims)
        self.fusion = Fusion2(2*dims[-1], dims[-1])

        self.detector = DetectorAttention(nhead=8, d_model=dims[-1], n_layers=1)
        self.av_head = lossAV(dims[-1])
        self.v_head = lossV(dims[-1])

        self.dino_loss = True
        if self.dino_loss:
            self.dino_head = DINOHead(in_dim=dims[-1], out_dim=dims[-1]*4, hidden_dim=dims[-1]*2)
            self.t_temp = 0.04   # teacher temperature
            self.s_temp = 0.1    # student temperature
            self.c_momentum = 0.9  # EMA momentum of the teacher center
            self.register_buffer("center", torch.zeros(1, dims[-1]*4))

        # Loss type used by pyasd.models.Criterion
        self.postprocess = 'softmax'

    def forward_visual_frontend(self, x):
        """Normalize the face crops and encode them.

        Args:
            x (torch.Tensor): Grayscale faces in [0, 1] of shape (B, T, 1, 112, 112).

        Returns:
            torch.Tensor: Visual embeddings of shape (B, T, C).
        """
        x = x[:, :, 0, :, :]
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        """Encode the MFCC features.

        Args:
            x (torch.Tensor): MFCCs of shape (B, 13, 4*T).

        Returns:
            torch.Tensor: Audio embeddings of shape (B, T, C).
        """
        x = x.unsqueeze(1)
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):
        """Fuse the audio and visual embeddings and run the temporal detector.

        Args:
            x1 (torch.Tensor): Audio embeddings of shape (B, T, C).
            x2 (torch.Tensor): Visual embeddings of shape (B, T, C).

        Returns:
            torch.Tensor: Audio-visual embeddings of shape (B, T, C).
        """
        x = self.fusion(x1, x2)
        x = self.detector(x)
        return x

    def forward_visual_backend(self,x):
        """Identity: the visual-only head works directly on the visual embeddings.

        Args:
            x (torch.Tensor): Visual embeddings of shape (B, T, C).

        Returns:
            torch.Tensor: The same tensor.
        """
        return x

    def forward(self, visualFeature, audioFeature):
        """Predict frame-level speaking activity for a face track.

        Args:
            visualFeature (torch.Tensor): Faces in [0, 1] of shape (B, T, 1, 112, 112).
            audioFeature (torch.Tensor): MFCCs of shape (B, 13, 4*T).

        Returns:
            dict: Always contains ``'pred_logits'`` (B, T, 2).
            In training mode it also contains:

            - ``'aux_video_outputs'``: ``[{'pred_logits': (B, T, 2)}]`` from the
              visual-only head.
            - ``'aux_dino_probs'``: ``{'student': log-probs, 'teacher': probs}``,
              each of shape (B*T, 4*C).

            In eval mode it contains ``'pred_scores'`` (B, T, 2), the softmax
            probabilities, where index 1 means "speaking".
        """
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        logits = self.av_head(outsAV)

        outputs = {'pred_logits': logits}

        if self.training:
            outsV = self.forward_visual_backend(visualEmbed)
            outputs['aux_video_outputs'] = [{'pred_logits': self.v_head(outsV)}]
            if self.dino_loss:
                s_out = self.dino_head(visualEmbed).flatten(0, 1)
                s_probs = F.log_softmax(s_out / self.s_temp, dim=-1)
                with torch.no_grad():
                    t_out = self.dino_head(outsAV).flatten(0, 1)
                    t_probs = F.softmax((t_out - self.center) / self.t_temp, dim=-1).detach()
                outputs['aux_dino_probs'] = {'student': s_probs , 'teacher': t_probs}
                self.update_center(t_out)
        else:
            scores = F.softmax(logits, dim=-1)
            outputs['pred_scores'] = scores

        return outputs

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update the teacher center with an exponential moving average.

        ``center = m * center + (1 - m) * mean(teacher_output)``

        Args:
            teacher_output (torch.Tensor): Teacher logits of shape (N, 4*C).
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.c_momentum + batch_center * (1 - self.c_momentum)

    def deploy(self):
        """Convert the model for inference (in place, idempotent).

        This switches to eval mode, fuses every RepVGG branch into a single
        convolution, and deletes the training-only modules (``v_head``,
        ``dino_head``, ``center``).

        Returns:
            LRXASDBase: ``self``.
        """
        self.eval()
        if not hasattr(self, 'deploy_'):
            for m in self.modules():
                if hasattr(m, "convert_to_deploy"):
                    m.convert_to_deploy()
            del self.v_head
            del self.dino_head
            del self.center
            self.deploy_ = True
        return self


#: LRX-ASD (default): encoder channels [32, 64, 128].
LRXASD = partial(LRXASDBase, dims=[32, 64, 128])
#: LRX-ASD-Small: encoder channels [16, 32, 64].
LRXASDSmall = partial(LRXASDBase, dims=[16, 32, 64])
#: LRX-ASD-Tiny: encoder channels [8, 16, 32].
LRXASDTiny = partial(LRXASDBase, dims=[8, 16, 32])
