"""
Modified from DEIM (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from  RT-DETR ()
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Training loop and evaluation routines (AVA-ActiveSpeaker and Columbia ASD).
"""
import os
import sys
import glob
import math
import tqdm
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

import torch

from pyasd.misc import dist_utils
from .logger import MetricLogger, SmoothedValue
from .get_ava_active_speaker_performance import run_evaluation

def train_one_epoch(
    model, criterion,
    data_loader, scaler, optimizer,
    device, epoch, max_norm, **kwargs
    ):
    """Train the model for one epoch.

    Args:
        model (nn.Module): Model returning a dict with ``'pred_logits'`` (and
            auxiliary outputs in training mode).
        criterion (Criterion): Loss module, see :class:`pyasd.models.Criterion`.
        data_loader (DataLoader): Loader over :class:`pyasd.data.train_loader`
            (``batch_size=1``; each item is already a mini-batch).
        scaler (torch.amp.GradScaler | None): Gradient scaler for AMP. ``None``
            trains in full precision.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str | torch.device): Training device.
        epoch (int): Current epoch (also sets the criterion temperature).
        max_norm (float): Gradient-clipping max norm. ``<= 0`` disables clipping.
        **kwargs: ``print_freq`` (int, default 10), ``writer`` (TensorBoard
            ``SummaryWriter``, optional) and ``ema`` (:class:`ModelEMA`, optional).

    Returns:
        dict[str, float]: Epoch averages of the logged meters (``loss``, each
        loss term and ``lr``).
    """
    model.train()
    criterion.train()
    criterion.set_epoch(epoch)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    writer = kwargs.get('writer', None)
    ema = kwargs.get('ema', None)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = {k: v[0].to(device) for k, v in batch.items()}
        global_step = epoch * len(data_loader) + i

        targets = {'labels': batch['labels']}

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(batch['video'], batch['audio'])
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        else:
            outputs = model(batch['video'], batch['audio'])
            loss_dict = criterion(outputs, targets)

            loss : torch.Tensor = sum(loss_dict.values())
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # update ema
        if ema is not None:
            ema.update(model)


        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_dict_reduced = {k: v.item() for k, v in loss_dict_reduced.items()}
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Outputs:")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print("  > {}: {}".format(k, v.isnan().any()))
                else:
                    print("  > {} is a {}, not a Tensor".format(k, type(v)))

            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluateAVA(model, data_loader, device, evalCsvSave, evalOrig, use_fp16, **kwargs):
    """Evaluate on AVA-ActiveSpeaker with the official mAP protocol.

    The frame-level speaking scores are written in the order of ``evalOrig``
    to ``evalCsvSave``, which is then scored with the official AVA script.

    Args:
        model (nn.Module): Model whose eval-mode output has ``'pred_scores'`` (1, T, 2).
        data_loader (DataLoader): Loader over :class:`pyasd.data.val_loader`.
        device (str | torch.device): Evaluation device.
        evalCsvSave (str): Output CSV with the predictions.
        evalOrig (str): Ground-truth CSV, e.g. ``csv/val_orig.csv``.
        use_fp16 (bool): Run inference under float16 autocast.
        **kwargs: ``print_freq`` (int, default 100).

    Returns:
        float: mAP in percent.
    """
    model.eval()
    all_preds = []

    print_freq = kwargs.get('print_freq', 100)

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = {k: v[0].to(device) for k, v in batch.items()}

        with torch.autocast(
            device_type=str(device), 
            enabled=use_fp16, 
            dtype=torch.float16):
            outputs = model(batch['video'], batch['audio'])
        scores = outputs['pred_scores'][0, :, 1]
        scores = scores.cpu().numpy()
        all_preds.extend(scores)

    # 3. Align and Save with Pandas
    # Create a DataFrame of predictions and sort it to match the original CSV order
    df_preds = pd.Series(all_preds)
    
    # Load metadata from original file
    evalRes = pd.read_csv(evalOrig)
    
    # Safety check: ensure lengths match
    if len(evalRes) != len(df_preds):
        print(f"Warning: Metadata ({len(evalRes)}) and Preds ({len(df_preds)}) size mismatch!")

    evalRes['score'] = df_preds
    evalRes['label'] = 'SPEAKING_AUDIBLE' # Required label for AVA script
    
    # Drop internal helper columns
    evalRes.drop(['label_id', 'instance_id'], axis=1, errors='ignore', inplace=True)
    evalRes.to_csv(evalCsvSave, index=False)

    # 4. Direct call to the evaluation logic
    mAP = run_evaluation(evalOrig, evalCsvSave)

    return mAP


def evaluateColumbia(trackPath, framesPath, labelsPath, scores, fps, isColumbia=False):
    """Evaluate frame-level ASD on Columbia ASD (or a dataset with the same layout).

    Every ground-truth face box is matched to the predicted face track with the
    highest overlap (> 0.5). The track score is smoothed over 5 frames and
    thresholded at 0.5. Per-speaker accuracy and F1 are printed, followed by
    their plain and sample-weighted averages.

    Args:
        trackPath (str): ``pywork/tracks.pckl`` written by the LR-ASD/TalkNet
            preprocessing pipeline.
        framesPath (str): ``pyframes`` folder with the extracted video frames.
        labelsPath (str): Glob of the per-speaker label files, e.g.
            ``col_labels/fusion/*.txt``.
        scores (list[numpy.ndarray]): Per-track, per-frame speaking scores, in
            the same order as the tracks.
        fps (float): Frame rate of the original video (labels are resampled to 25 fps).
        isColumbia (bool): Columbia-specific label format (square boxes) and
            half-resolution coordinates.
    """
    # Load tracks
    fil = open(trackPath, 'rb')
    tracks = pickle.load(fil)    

    # Get speakers name
    names = glob.glob(labelsPath)
    names = [os.path.splitext(os.path.basename(f))[0] for f in names]

    # Initiliase dictionaries
    predictionSet = {}
    for name in names:
        predictionSet[name] = [[], []]

    # Retrieve real bounding boxes
    dictGT = {}
    txtFiles = glob.glob("%s"%labelsPath)
    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = file.split('/')[-1][:-4]
        for line in lines:
            data = line.split('\t')
            frame = int(int(data[0]) / fps * 25)
            x1 = int(float(data[1]))
            y1 = int(float(data[2]))
            if isColumbia:
                x2 = x1 + int(float(data[3]))
                y2 = y1 + int(float(data[3]))
                gt = int(float(data[4]))
            else:
                x2 = x1 + int(float(data[3]))
                y2 = y1 + int(float(data[4]))
                gt = int(float(data[5]))
            if frame in dictGT:
                dictGT[frame].append([x1,y1,x2,y2,gt,idName])
            else:
                dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]

    # Load frames filenames
    flist = glob.glob(os.path.join(framesPath, '*.jpg')) 
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]                
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = np.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
    
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        if fidx in dictGT: # This frame has label
            for gtThisFrame in dictGT[fidx]: # What this label is ?
                faceGT = gtThisFrame[0:4]
                labelGT = gtThisFrame[4]
                idGT = gtThisFrame[5]
                ious = []
                for face in faces[fidx]: # Find the right face in my result
                    if isColumbia:
                        faceLocation = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
                    else:
                        faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
                    iou = bb_intersection_over_union(faceLocation, faceGT, evalCol = True)
                    if iou > 0.5:
                        ious.append([iou, face['score']])
                if len(ious) > 0: # Find my result
                    ious.sort()
                    labelPredict = ious[-1][1]
                else:                   
                    labelPredict = 0

                predictionSet[idGT][0].append(labelPredict)
                predictionSet[idGT][1].append(labelGT)

    total_f1_weighted = 0.0
    total_acc_weighted = 0.0
    F1 = 0
    ACC = 0 
    N = 0
    total_samples = 0

    names.sort()
    print(f"{'Name':<15} | {'Accuracy':>10} | {'F1 Score':>10} | {'Samples':>8}")
    print("-" * 55)
    for name in names:
        # 1. Extraction and Thresholding
        # Assuming predictionSet[name][0] are scores and [1] are labels
        scores = np.array(predictionSet[name][0])
        labels = np.array(predictionSet[name][1])
        preds = (scores > 0.5).astype(np.int64)
        num_objects = len(labels)
        
        # 2. Metrics
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)

        # Following the TalkNet / LR-ASD Columbia protocol, speaker "abbas" is
        # excluded (only Bell, Boll, Lieb, Long and Sick are reported).
        if name == 'abbas':
            continue
            
        # 3. Printing with f-strings (Modern Python approach)
        print(f"{name:<15} | {100 * acc:>9.2f}% | {100 * f1:>9.2f}% | {num_objects:>8}")
        
        # 4. Weighted Average Calculation
        # Weight = number of objects for this specific person
        total_f1_weighted += f1 * num_objects
        total_acc_weighted += acc * num_objects
        total_samples += num_objects

        F1 += f1
        ACC += acc
        N += 1

    # Final Summary
    if total_samples > 0:
        avg_f1 = (total_f1_weighted / total_samples) * 100
        avg_acc = (total_acc_weighted / total_samples) * 100
        orig_avg_f1 = F1 / N * 100
        orig_avg_acc = ACC / N * 100
        print("-" * 55)
        print(f"{'TOTAL NUMBER OF SAMPLES:':<25} {total_samples:>11}")
        print(f"{'WEIGHTED AVG F1:':<25} {avg_f1:>10.2f}%")
        print(f"{'WEIGHTED AVG ACC:':<25} {avg_acc:>10.2f}%")
        print(f"{'AVG F1:':<25} {orig_avg_f1:>10.2f}%")
        print(f"{'AVG ACC:':<25} {orig_avg_acc:>10.2f}%")
        print()


def bb_intersection_over_union(boxA, boxB, evalCol = False):
    """Overlap between two ``[x1, y1, x2, y2]`` boxes.

    Args:
        boxA (list[float]): Predicted box.
        boxB (list[float]): Ground-truth box.
        evalCol (bool): If True, normalize the intersection by the area of
            ``boxA`` only (Columbia protocol) instead of by the union.

    Returns:
        float: IoU (or intersection over the area of ``boxA``).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
