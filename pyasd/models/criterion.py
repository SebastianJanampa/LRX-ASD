"""Training criterion shared by all ASD models."""

import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Criterion(nn.Module):
    """Compute the weighted ASD losses from a model's output dict.

    Losses computed:

    - the main audio-visual loss on ``outputs['pred_logits']``, weighted by
      ``weight_dict['av_loss']``;
    - auxiliary audio-visual losses on ``outputs['aux_outputs']`` (if present),
      weighted by ``weight_dict['av_loss']``;
    - visual-only losses on ``outputs['aux_video_outputs']`` (if present),
      weighted by ``weight_dict['v_loss']``;
    - the DINO self-distillation loss on ``outputs['aux_dino_probs']`` (if
      present), weighted by ``weight_dict['loss_dino']``.

    Args:
        weight_dict (dict): Loss weights. Needs the keys ``'av_loss'`` and
            ``'v_loss'``, plus ``'loss_dino'`` for models with a DINO head.
        losses (list[str]): Classification losses to apply, from ``'softmax'``
            (temperature-scaled BCE on the speaking probability) and ``'focal'``.
        num_classes (int): Number of classes (2: not speaking / speaking).
    """

    def __init__(self, weight_dict, losses, num_classes):
        super(Criterion, self).__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_classes = num_classes
        self.alpha = -1.0  # focal loss alpha (-1 disables class weighting)
        self.gamma = 2.0

        self.init_r = 1.3
        self.cur_r = 1.3

    def set_epoch(self, epoch):
        """Update the softmax temperature: ``r = max(1.3 - 0.02 * epoch, 0.1)`` (as in LR-ASD).

        Args:
            epoch (int): Current (0-based) epoch.
        """
        self.cur_r = max(self.init_r - (0.02) * (epoch), 0.1)

    def loss_focal_labels(self, outputs, targets):
        """Sigmoid focal loss on the logits.

        Args:
            outputs (dict): Contains ``'pred_logits'`` (B, T, num_classes).
            targets (dict): Contains ``'labels'`` (B, T) with integer labels.

        Returns:
            dict: ``{'loss_cls': scalar tensor}``.
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        target_classes =  targets['labels']
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits.float(), target.to(float), self.alpha, self.gamma, reduction="mean"
        )

        return {"loss_cls": loss}

    def loss_softmax_labels(self, outputs, targets):
        """Binary cross-entropy on the temperature-scaled softmax probability of "speaking".

        Args:
            outputs (dict): Contains ``'pred_logits'`` (B, T, 2).
            targets (dict): Contains ``'labels'`` (B, T) with 0/1 labels.

        Returns:
            dict: ``{'loss_cls': scalar tensor}``.
        """
        assert "pred_logits" in outputs
        src_logits = (outputs["pred_logits"] / self.cur_r).softmax(dim=-1)[:, :, 1].float()
        target_classes =  targets['labels'].float()

        loss = F.binary_cross_entropy(src_logits, target_classes)

        return {"loss_cls": loss}

    def loss_dino(self, outputs, targets):
        """DINO cross-entropy between teacher (audio-visual) and student (visual) distributions.

        Args:
            outputs (dict): Contains ``'aux_dino_probs'``, a dict with
                ``'teacher'`` (probabilities) and ``'student'`` (log-probabilities),
                each of shape (N, K).
            targets (dict): Unused.

        Returns:
            dict: ``{'loss_dino': scalar tensor}``.
        """
        assert "aux_dino_probs" in outputs
        t_probs = outputs['aux_dino_probs']['teacher']
        s_probs = outputs['aux_dino_probs']['student']

        loss = -(t_probs * s_probs).sum(dim=-1).mean()
        return {'loss_dino': loss}

    def get_loss(self, loss, outputs, targets):
        """Dispatch to the classification loss named ``loss`` (``'focal'`` or ``'softmax'``).

        Args:
            loss (str): Loss name.
            outputs (dict): Model outputs.
            targets (dict): Targets with key ``'labels'``.

        Returns:
            dict: The loss dict returned by the selected loss.
        """
        loss_map = {
            "focal": self.loss_focal_labels,
            "softmax": self.loss_softmax_labels,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """Compute all the weighted losses.

        Args:
            outputs (dict): Model outputs (see the model's ``forward``).
            targets (dict): Contains ``'labels'`` (B, T).

        Returns:
            dict[str, torch.Tensor]: Weighted scalar losses, e.g. ``loss_cls``,
            ``loss_dino_aux`` and ``loss_cls_only_video_aux_0``. The total loss is
            their sum.
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets)
            l_dict = {k: l_dict[k] * self.weight_dict['av_loss'] for k in l_dict}
            losses.update(l_dict)

        if "aux_dino_probs" in outputs:
            l_dict = self.loss_dino(outputs, targets)

            l_dict = {
                k: l_dict[k] * self.weight_dict[k] for k in l_dict 
            }
            l_dict = {k + "_aux": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets)

                    l_dict = {
                        k: l_dict[k] * self.weight_dict['av_loss'] for k in l_dict 
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "aux_video_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_video_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets)

                    l_dict = {
                        k: l_dict[k] * self.weight_dict['v_loss'] for k in l_dict
                    }
                    l_dict = {k + f"_only_video_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
