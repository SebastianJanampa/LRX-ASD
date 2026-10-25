"""Train an ASD model on AVA-ActiveSpeaker.

Example:
    python tools/train.py --model lrxasd --data-path AVADataPath \
        --output-dir outputs/train/lrxasd --amp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyasd.engine.model import MODELS, Model  # noqa: E402


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lrxasd', choices=list(MODELS), help='architecture to train')
    parser.add_argument('--data-path', required=True, help='AVADataPath root (csv/, clips_audios/, clips_videos/)')
    parser.add_argument('--val-split', default='val', help="validation split; use 'none' to disable validation")
    parser.add_argument('--val-orig-csv', default=None,
                        help='ground-truth CSV of the validation split (default: <data-path>/csv/<val-split>_orig.csv)')
    parser.add_argument('--output-dir', default='outputs/train/exp1', help='where checkpoints and logs are written')
    parser.add_argument('--exist-ok', action='store_true', help='reuse --output-dir even if it exists')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    # Optimization
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='AdamW learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='AdamW weight decay')
    parser.add_argument('--lr-gamma', type=float, default=0.95, help='per-epoch LR decay factor')
    parser.add_argument('--max-frames', type=int, default=4000, help='total number of frames per mini-batch')
    parser.add_argument('--num-workers', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='mixed-precision training')
    parser.add_argument('--no-ema', action='store_true', help='disable the EMA of the weights')
    parser.add_argument('--clip-max-norm', type=float, default=-1, help='gradient clipping (<= 0 disables it)')
    # Loss weights
    parser.add_argument('--av-loss', type=float, default=1.0, help='weight of the audio-visual loss')
    parser.add_argument('--v-loss', type=float, default=0.5, help='weight of the visual-only auxiliary loss')
    # Logging / checkpoints
    parser.add_argument('--print-freq', type=int, default=300, help='training log frequency (iterations)')
    parser.add_argument('--print-freq-val', type=int, default=2000, help='validation log frequency (iterations)')
    parser.add_argument('--checkpoint-freq', type=int, default=15, help='save an extra checkpoint every N epochs')
    parser.add_argument('--resume', default=None, help='training checkpoint to resume from')
    return parser.parse_args()


def main():
    """Build the model and launch training."""
    args = parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    val_split = None if args.val_split.lower() == 'none' else args.val_split
    val_orig_csv = args.val_orig_csv
    if val_split and val_orig_csv is None:
        val_orig_csv = os.path.join(args.data_path, 'csv', f'{val_split}_orig.csv')

    model = Model(args.model, device=args.device)
    model.train(
        dataset_path=args.data_path,
        val_dataset_type=val_split,
        val_origCsv=val_orig_csv,
        output_dir=args.output_dir,
        exist_ok=args.exist_ok,
        max_number_frames=args.max_frames,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_gamma=args.lr_gamma,
        use_ema=not args.no_ema,
        total_epochs=args.epochs,
        amp=args.amp,
        clip_max_norm=args.clip_max_norm,
        print_freq=args.print_freq,
        print_freq_val=args.print_freq_val,
        checkpoint_freq=args.checkpoint_freq,
        resume=args.resume,
        audio_video_loss=args.av_loss,
        video_loss=args.v_loss,
    )


if __name__ == '__main__':
    main()
