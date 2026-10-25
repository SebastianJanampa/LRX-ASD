"""High-level :class:`Model` API for training, evaluation and inference of ASD models."""

import os
import re
import cv2
import glob
import copy
import time
import json
import math
import tqdm
import numpy as np
import python_speech_features

from scipy.io import wavfile
from pathlib import Path
from datetime import datetime, timedelta

import torch

from pyasd.misc import dist_utils
from pyasd.engine.engine import evaluateColumbia
from pyasd.models import TalkNet, LRASD, LRXASD, LRXASDSmall, LRXASDTiny, LightASD


#: Registry of the available architectures (name -> constructor).
MODELS = {
    'talknet': TalkNet,
    'lrasd': LRASD,
    'lrxasd': LRXASD,
    'lrxasd_small': LRXASDSmall,
    'lrxasd_tiny': LRXASDTiny,
    'lightasd': LightASD,
}

class Model(object):
    """Wrapper around an ASD network that handles training, evaluation and inference.

    Example:
        >>> model = Model('lrxasd')
        >>> model.load_weights('lrx_asd.model')
        >>> scores = model(dataPath='demo/pycrop')  # per-track, per-frame scores

    Args:
        model (str): Architecture name, one of the keys of :data:`MODELS`
            (``'talknet'``, ``'lrasd'``, ``'lightasd'``, ``'lrxasd'``).
        device (str, optional): Device to use. Defaults to ``'cuda'`` when
            available, otherwise ``'cpu'``.
    """

    def __init__(self, model, device=None):
        assert model in MODELS, f'{model} is not available\n' + \
        f'the available models are: {MODELS.keys()}'

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device 
        
        self.model = MODELS[model]() # use for training/eval/inference
        self.model = self.model.to(self.device)

    def model_analysis(self, input_size=[112, 112], print_detailed=False):
        """Print the FLOPs, MACs and parameter count of the deployed model for a 1-frame input.

        Requires the optional ``calflops`` package (``pip install -e .[dev]``).

        Args:
            input_size (int | list[int]): Face crop size (H, W).
            print_detailed (bool): Print a per-layer breakdown.
        """
        from calflops import calculate_flops
        
        if isinstance(input_size, (int, float)):
            input_size = int(input_size)
            input_size = [input_size, input_size]
        video_input_shape = tuple([3, 1, 1] + input_size)
        audio_input_shape = tuple([3, 13, 4])

        print(f"FLOPs Analysis - Input Shape of {video_input_shape}")

        model_for_info = copy.deepcopy(self.model)
        if hasattr(model_for_info, 'deploy'):
            model_for_info = model_for_info.deploy()
        else:
            model_for_info.eval()

        video_input = torch.rand(video_input_shape)
        audio_input = torch.rand(audio_input_shape)

        flops, macs, _ = calculate_flops(model=model_for_info,
                                        kwargs={'visualFeature':video_input, 'audioFeature':audio_input},
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=print_detailed,
                                        )
        params = sum(p.numel() for p in model_for_info.parameters())
        print("Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params))

        del model_for_info

    @torch.no_grad()
    def __call__(self, dataPath, videoPath=None, durationSet={1:1}, use_fp16=False,
        duration_detailed=False, evaluate=False, gtPath=None, isColumbia=False):
        """Run inference on pre-cropped face tracks (and optionally evaluate them).

        ``dataPath`` must be the ``pycrop`` folder produced by the LR-ASD /
        TalkNet preprocessing pipeline: one ``<track>.avi`` (face crops) and one
        ``<track>.wav`` (16 kHz audio) per face track. Each track is cut into
        chunks of ``duration`` seconds, scored chunk by chunk, and the scores of
        all durations are averaged with the given weights.

        Args:
            dataPath (str): Path to the ``pycrop`` folder.
            videoPath (str, optional): Original video. Only used during
                evaluation of non-Columbia data, to read its frame rate.
            durationSet (dict[int, float]): ``{chunk length in seconds: weight}``,
                e.g. ``{1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}``.
            use_fp16 (bool): Run the network under float16 autocast (CUDA only).
            duration_detailed (bool): Also evaluate each duration separately
                (only used when ``evaluate=True``).
            evaluate (bool): Compute accuracy/F1 against ``gtPath`` with
                :func:`pyasd.engine.engine.evaluateColumbia`. Requires the
                sibling ``pyframes`` and ``pywork`` folders.
            gtPath (str, optional): Ground-truth folder with ``fusion/*.txt`` label files.
            isColumbia (bool): Use the Columbia ASD conventions (29.97 fps video,
                25 fps crops, square boxes).

        Returns:
            list[numpy.ndarray]: One array of per-frame speaking probabilities per
            track, sorted by track file name.
        """
        if evaluate:
            assert gtPath, 'if evaluate=True, you must provide gtPath'
            for folder in ('pyframes', 'pywork'):
                path = dataPath.replace('pycrop', folder)
                assert os.path.exists(path), f'{path} does not exist'

        if hasattr(self.model, 'deploy'):
            self.model = self.model.deploy()
        else:
            self.model.eval()
        files = glob.glob("%s/*.avi"%dataPath)
        files.sort()

        if isColumbia:
            fps_video = 29.97
        elif evaluate:
            assert videoPath, 'if evaluate=True on non-Columbia data, you must provide videoPath'
            main_video = cv2.VideoCapture(videoPath)
            fps_video = main_video.get(cv2.CAP_PROP_FPS)
            main_video.release()

        finalScores = []
        durationScores = {i: [] for i in durationSet} if duration_detailed else None
        for file in tqdm.tqdm(files, total = len(files)):
            fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
            
            # Load video 
            video = cv2.VideoCapture(os.path.join(dataPath, fileName + '.avi'))
            videoFeature, fps = [], round(video.get(cv2.CAP_PROP_FPS))
            while video.isOpened():
                ret, frames = video.read()
                if ret == True:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (112, 112))
                    videoFeature.append(face[None])
                else:
                    break
            video.release()
            videoFeature = np.stack(videoFeature, axis=0)

            # Load audio
            # video-audio ratio -> 1:4
            if isColumbia:
                fps = 25

            hzAudio = 4 * fps # Hertz
            _, audio = wavfile.read(os.path.join(dataPath, fileName + '.wav'))
            audioFeature = python_speech_features.mfcc(
                audio, 16000, numcep = 13, 
                # winlen and winstep controls the length of the spectogram
                winlen = 0.025 * 25 / fps, winstep = 0.01 * 25 / fps)

            # From array to tensor
            dtype = torch.float
            videoFeature = torch.tensor(videoFeature, device=self.device, dtype=dtype).unsqueeze(0) / 255
            audioFeature = torch.tensor(audioFeature, device=self.device, dtype=dtype).unsqueeze(0).transpose(1, 2)
            length = min((audioFeature.shape[2] - audioFeature.shape[2] % 4) / hzAudio, videoFeature.shape[1])
            audioFeature = audioFeature[:, :, :int(round(length * hzAudio))]
            videoFeature = videoFeature[:, :int(round(length * fps))]
            # Detection happens here
            avgScore = []
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                curDurationScores = []
                for i in range(batchSize):
                    inputA = audioFeature[:, :, i * duration * hzAudio: (i+1) * duration * hzAudio]
                    inputV = videoFeature[:, i * duration * fps: (i+1) * duration * fps]
                    with torch.autocast(device_type=str(self.device), dtype=torch.float16,
                                        enabled=use_fp16 and str(self.device).startswith('cuda')):
                        outs = self.model(inputV, inputA)
                    score = outs['pred_scores']
                    score = score[..., 1].flatten().float().cpu().numpy()
                    curDurationScores.extend(score)
                avgScore.append(curDurationScores)
                if durationScores:
                    durationScores[duration].append(curDurationScores)
            avgScore = np.stack(avgScore, axis=0)
            weights = list(durationSet.values())
            avgScore = np.average(avgScore, axis=0, weights=weights)
            finalScores.append(avgScore)

        if evaluate:
            # Evaluate the results following the Columbia dataset protocol
            trackPath = dataPath.replace('pycrop', 'pywork') + '/tracks.pckl'
            framesPath = dataPath.replace('pycrop', 'pyframes')
            labelsPath = gtPath + '/fusion/*.txt'

            print("#"*25 + f"{' Results ':^30}" + "#"*25)
            keys_str = ", ".join(map(str, durationSet.keys()))
            vals_str = ", ".join(map(str, durationSet.values()))
            print(f'Durations (in seconds): {keys_str:<20} | Weights: {vals_str}')
            evaluateColumbia(trackPath, framesPath, labelsPath, finalScores, fps_video, isColumbia)

            if durationScores:
                for k, v in durationSet.items():
                    print(f'Durations (in seconds): {k:<20} | Weights: {v}')
                    evaluateColumbia(trackPath, framesPath, labelsPath, durationScores[k], fps_video, isColumbia)

        return finalScores

    @staticmethod
    def check_folder(output_dir, exist_ok=False):
        """
        Creates an output directory, incrementing a numeric suffix if the path already exists.

        For example, if 'outputs/train/exp1' exists, it will create 'outputs/train/exp2'.
        If 'outputs/train/exp' is given, it will start by checking for 'outputs/train/exp1'.
        
        Args:
            output_dir (str): The desired directory path, e.g., 'outputs/train/exp1'.
            exist_ok (bool): If True, it will use the provided `output_dir` as is,
                             creating it if it doesn't exist, without searching for a new path.
        
        Returns:
            str: The path to the created directory.
        """
        if exist_ok:
            os.makedirs(output_dir, exist_ok=True)
            return output_dir

        # Separate the base path from any existing numeric suffix
        # e.g., 'outputs/train/exp1' -> base='outputs/train/exp', num=1
        match = re.match(r'(.*?)(\d*)$', output_dir)
        base, num_str = match.groups()
        
        # If the original path had no number, start counting from 1.
        # Otherwise, start from the number that was found.
        counter = int(num_str) if num_str else 1
        
        current_dir = f"{base}{counter}"

        # Increment the counter until a non-existent path is found
        while os.path.exists(current_dir):
            counter += 1
            current_dir = f"{base}{counter}"
            
        os.makedirs(current_dir, exist_ok=True)
        return current_dir

    def create_dataloader(self, 
        # Dataset arguments
        dataset_path, 
        split_set, 
        max_number_frames,
        num_workers,
        ):
        """Build an AVA-ActiveSpeaker ``DataLoader``.

        Args:
            dataset_path (str): ``AVADataPath`` root containing ``csv/``,
                ``clips_audios/`` and ``clips_videos/``.
            split_set (str): ``'train'``, ``'val'`` or ``'test'``. Only
                ``'train'`` uses augmentation, length-sorted mini-batches and shuffling.
            max_number_frames (int): Frames per training mini-batch. Use 1 for
                evaluation: one clip per forward pass, as needed to replicate the
                AVA results.
            num_workers (int): Number of ``DataLoader`` workers.

        Returns:
            torch.utils.data.DataLoader: Loader with ``batch_size=1`` (each item
            is already a mini-batch).
        """
        from pyasd.data import train_loader, val_loader 
        from torch.utils.data import DataLoader

        training = split_set=='train'
        csv_file = f"{dataset_path}/csv/{split_set}_loader.csv"
        audio_path = f"{dataset_path}/clips_audios/{split_set}"
        video_path = f"{dataset_path}/clips_videos/{split_set}"

        if training:
            dataset = train_loader(
                trialFileName = csv_file,
                audioPath=audio_path,
                visualPath=video_path,
                batchSize=max_number_frames
                )
        else:
            dataset = val_loader(
                trialFileName = csv_file,
                audioPath=audio_path,
                visualPath=video_path,
                )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=training,
            num_workers=num_workers,
            pin_memory=True if self.device=='cuda' else False,
            prefetch_factor = 2
            )

        return dataloader

    def train(
        self,
        dataset_path,
        val_dataset_type=None,
        val_origCsv=None,
        output_dir='outputs/train/exp1',
        exist_ok=False,
        # Dataloader params
        max_number_frames=2000,
        num_workers=16,
        # Optimizer params
        lr=1e-3,
        weight_decay=0.01,
        # LR Scheduler
        lr_gamma=0.95,
        # EMA params
        use_ema=True,
        ema_decay=0.9999,
        ema_warmups=1000,
        # Training schedule
        total_epochs=30,
        amp=False,
        # Extra training params
        clip_max_norm=-1,
        print_freq=1000, 
        print_freq_val=100,
        checkpoint_freq=15,
        resume=None,
        # Criterion params
        audio_video_loss=1.0,
        video_loss=0.5,
        ):
        """Train on AVA-ActiveSpeaker, validating (mAP) after every epoch.

        Checkpoints are written to ``output_dir``: ``last.pth`` every epoch,
        ``checkpointXXXX.pth`` every ``checkpoint_freq`` epochs, and ``best.pth``
        (best validation mAP). ``log.txt`` (JSON lines) and TensorBoard logs
        (``summary/``, if tensorboard is installed) are written there too.

        Args:
            dataset_path (str): ``AVADataPath`` root.
            val_dataset_type (str, optional): Validation split (e.g. ``'val'``).
                ``None`` disables validation.
            val_origCsv (str, optional): Ground-truth CSV of the validation split
                (e.g. ``AVADataPath/csv/val_orig.csv``). Needed for validation.
            output_dir (str): Output folder. If it exists and ``exist_ok`` is
                False, a numeric suffix is incremented (``exp1`` -> ``exp2``).
            exist_ok (bool): Reuse ``output_dir`` even if it exists.
            max_number_frames (int): Total frames per training mini-batch.
            num_workers (int): ``DataLoader`` workers.
            lr (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay.
            lr_gamma (float): Per-epoch multiplicative LR decay (``StepLR``).
            use_ema (bool): Keep an EMA of the weights (used for validation and
                saved under the ``'ema'`` key).
            ema_decay (float): EMA decay.
            ema_warmups (int): EMA warm-up iterations.
            total_epochs (int): Number of training epochs.
            amp (bool): Mixed-precision training (and fp16 validation).
            clip_max_norm (float): Gradient clipping max norm. ``<= 0`` disables clipping.
            print_freq (int): Training log frequency (iterations).
            print_freq_val (int): Validation log frequency (iterations).
            checkpoint_freq (int): Save an extra checkpoint every N epochs.
            resume (str, optional): Checkpoint (path or URL) to resume from.
            audio_video_loss (float): Weight of the audio-visual loss.
            video_loss (float): Weight of the visual-only auxiliary loss.
        """
        from pyasd.optim import AdamW, GradScaler, ModelEMA
        from pyasd.engine import train_one_epoch, evaluateAVA
        from pyasd.models import Criterion

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            print("Warning: TensorBoard not found. Training logs will not be saved.")
            SummaryWriter = None # Set it to None if not available

        self.model = self.model.to(self.device)
        if use_ema:
            self.ema = ModelEMA(model=self.model, decay=ema_decay, warmups=ema_warmups)
        else:
            self.ema = None 
            
        # Datasets
        print("Loading training dataset")
        self.train_dataloader = self.create_dataloader(
            dataset_path, 'train', max_number_frames, num_workers)

        if val_dataset_type:
            assert val_origCsv
            print("Loading validation dataset")
            # max_number_frames=1: one clip per forward pass (needed to replicate AVA results)
            self.val_dataloader = self.create_dataloader(
                dataset_path, val_dataset_type, 1, num_workers)
            self.val_origCsv = val_origCsv
        else:
            self.val_dataloader = self.val_origCsv = None

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size = 1, gamma=lr_gamma)

        # Criterion
        weight_criterion = {'av_loss': audio_video_loss, 'v_loss': video_loss}
        if hasattr(self.model, 'dino_loss') and self.model.dino_loss:
            weight_criterion['loss_dino'] = 0.1
        loss_type = self.model.postprocess
        assert loss_type in ['focal', 'softmax'], f'loss type {loss_type} is not supported'
        self.criterion = Criterion(weight_criterion, [loss_type], num_classes=2)

        try:
            self.model_analysis()
        except ImportError:
            print("Warning: calflops not found, skipping FLOPs analysis (pip install calflops).")

        # AMP training
        if amp:
            self.scaler = GradScaler(self.device, enabled=True)
        else:
            self.scaler = None

        # Saving folders
        self.output_dir = Path(self.check_folder(output_dir, exist_ok))
        if SummaryWriter:
            self.writer = SummaryWriter(self.output_dir / 'summary')
        else:
            self.writer = None

        # Resuming training (optional)
        self.last_epoch = -1
        self.best_map = -1
        self.best_epoch = -1
        if resume:
            print(f'Resume checkpoint from {resume}')
            self.load_resume_state(resume)

        print("-"*42 + "Start training" + "-"*43)
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_frozen_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_frozen_parameters}')

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, total_epochs):
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.scaler,
                self.optimizer,
                self.device,
                epoch, 
                max_norm=clip_max_norm, 
                print_freq=print_freq, 
                ema=self.ema, 
                writer=self.writer
            )
            self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint every `checkpoint_freq` epochs
                if (epoch + 1) % checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            if self.val_dataloader:
                module = self.ema.module if self.ema else self.model
                cur_map = evaluateAVA(
                    module,
                    self.val_dataloader,
                    self.device,
                    f'{self.output_dir}/csv_results.csv',
                    self.val_origCsv,
                    use_fp16=amp,
                    print_freq=print_freq_val,
                )

                if self.writer:
                    self.writer.add_scalar('Test/mAP', cur_map, epoch)

                if self.best_map < cur_map:
                    self.best_map = cur_map
                    self.best_epoch = epoch
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
            
            print("  > Best Average Precision: %2.2f%%"%(self.best_map))
            print("  > Best Epoch: %2.0f"%(self.best_epoch))


            log_stats = {f'train_{k}': v for k, v in train_stats.items()}
            if self.val_dataloader:
                log_stats.update({'test_map': cur_map})
            log_stats.update({'epoch': epoch, 'n_parameters': n_parameters})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def validateAVA(
        self,
        dataset_path,
        val_dataset_type,
        val_origCsv,
        output_dir='outputs/val/exp1',
        exist_ok=False,
        weights=None,
        resume=None,
        use_fp16=True,
        # Dataloader params
        num_workers=16,
        print_freq=2000
        ):
        """Evaluate on an AVA-ActiveSpeaker split with the official mAP protocol.

        Args:
            dataset_path (str): ``AVADataPath`` root.
            val_dataset_type (str): Split to evaluate (``'val'`` or ``'test'``).
            val_origCsv (str): Ground-truth CSV of that split (e.g. ``csv/val_orig.csv``).
            output_dir (str): Folder where ``csv_results.csv`` is written.
            exist_ok (bool): Reuse ``output_dir`` even if it exists.
            weights (str, optional): Model weights to load with :meth:`load_weights`.
            resume (str, optional): Full training checkpoint to load with
                :meth:`load_resume_state`.
            use_fp16 (bool): Run inference under float16 autocast.
            num_workers (int): ``DataLoader`` workers.
            print_freq (int): Log frequency (iterations).

        Returns:
            float: mAP in percent.
        """
        from pyasd.engine import evaluateAVA

        self.model = self.model.to(self.device)
        self.ema = None

        # Datasets. max_number_frames=1: one clip per forward pass, which is
        # required to replicate the AVA results (no multi-clip batching).
        self.val_dataloader = self.create_dataloader(
            dataset_path, val_dataset_type, 1, num_workers)
        self.val_origCsv = val_origCsv


        # Saving folders
        self.output_dir = Path(self.check_folder(output_dir, exist_ok))

        if weights:
            self.load_weights(weights)
        if resume:
            print(f'Resume checkpoint from {resume}')
            self.load_resume_state(resume)

        module = self.ema.module if self.ema else self.model
        cur_map = evaluateAVA(
            module,
            self.val_dataloader,
            self.device,
            f'{self.output_dir}/csv_results.csv',
            self.val_origCsv,
            use_fp16=use_fp16,
            print_freq=print_freq,
        )
        return cur_map

    def state_dict(self):
        """Collect the state of every attribute that has a ``state_dict`` (model, ema, optimizer, ...).

        Returns:
            dict: Full training state, plus ``date``, ``last_epoch`` and ``best_map``.
        """
        state = {}
        state['date'] = datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch
        state['best_map'] = self.best_map

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def load_state_dict(self, state):
        """Restore the state produced by :meth:`state_dict` (missing/unexpected keys are reported).

        Args:
            state (dict): Full training state.
        """
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')
        if 'best_map' in state:
            self.best_map = state['best_map']
            print('Load best_map')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                try:
                    missing_keys, unexpected_keys = v.load_state_dict(state[k])
                except:
                    v.load_state_dict(state[k])
                    missing_keys = unexpected_keys = None

                perfect_loading = True
                if missing_keys:
                    perfect_loading = False
                    print("The following parameters are missing:")
                    for k_ in missing_keys:
                        print(f"  > {k_}")
                if unexpected_keys:
                    perfect_loading = False
                    print("The following parameters are not in model:")
                    for k_ in unexpected_keys:
                        print(f"  > {k_}")
                if perfect_loading:
                    print(f'Successfully loaded {k}.state_dict')


            if hasattr(v, 'load_state_dict') and k not in state:
                if k == 'ema':
                    model = getattr(self, 'model', None)
                    if model is not None:
                        ema = dist_utils.de_parallel(v)
                        model_state_dict = remove_module_prefix(model.state_dict())
                        ema.load_state_dict({'module': model_state_dict})
                        print(f'Load {k}.state_dict from model.state_dict')
                else:
                    print(f'Not load {k}.state_dict')

    def load_resume_state(self, path: str):
        """Load a full training checkpoint (model, EMA, optimizer, ...) to resume training.

        Args:
            path (str): Local path or URL of the checkpoint.
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=False)

        # state['model'] = remove_module_prefix(state['model'])
        self.load_state_dict(state)

    def load_weights(self, weights_path):
        """Load model weights only.

        Accepts either a released weight file (``{'model': state_dict}``) or a
        training checkpoint. For a training checkpoint, the EMA weights
        (``state['ema']['module']``) are used when present.

        Args:
            weights_path (str): Path to the ``.model`` / ``.pth`` file.
        """
        state = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'ema' in state:
            model_state_dict = state['ema']['module']
        else:
            model_state_dict = state['model']

        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict)
        perfect_loading = True
        if missing_keys:
            perfect_loading = False
            print("The following parameters are missing:")
            for k_ in missing_keys:
                print(f"  > {k_}")
        if unexpected_keys:
            perfect_loading = False
            print("The following parameters are not in model:")
            for k_ in unexpected_keys:
                print(f"  > {k_}")
        if perfect_loading:
            print('Successfully loaded model weights')

def remove_module_prefix(state_dict):
    """Strip the ``module.`` prefix that (Distributed)DataParallel adds to parameter names.

    Args:
        state_dict (dict): State dict, possibly with ``module.`` prefixes.

    Returns:
        dict: State dict without the prefixes.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
