"""Evaluate an ASD model on AVA-ActiveSpeaker (official frame-level mAP).

Example:
    python tools/eval_ava.py --model lrxasd --weights weights/lrx_asd.model \
        --data-path AVADataPath
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyasd.engine.model import MODELS, Model  # noqa: E402


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lrxasd', choices=list(MODELS), help='architecture to evaluate')
    parser.add_argument('--weights', required=True, help='model weights (.model) or training checkpoint (.pth)')
    parser.add_argument('--data-path', required=True, help='AVADataPath root (csv/, clips_audios/, clips_videos/)')
    parser.add_argument('--split', default='val', help='split to evaluate')
    parser.add_argument('--orig-csv', default=None,
                        help='ground-truth CSV (default: <data-path>/csv/<split>_orig.csv)')
    parser.add_argument('--output-dir', default='outputs/val/exp1', help='where csv_results.csv is written')
    parser.add_argument('--exist-ok', action='store_true', help='reuse --output-dir even if it exists')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    parser.add_argument('--fp32', action='store_true', help='disable float16 inference')
    parser.add_argument('--num-workers', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--print-freq', type=int, default=2000, help='log frequency (iterations)')
    return parser.parse_args()


def main():
    """Load the model and report the mAP."""
    args = parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    orig_csv = args.orig_csv or os.path.join(args.data_path, 'csv', f'{args.split}_orig.csv')

    model = Model(args.model, device=args.device)
    mAP = model.validateAVA(
        dataset_path=args.data_path,
        val_dataset_type=args.split,
        val_origCsv=orig_csv,
        output_dir=args.output_dir,
        exist_ok=args.exist_ok,
        weights=args.weights,
        use_fp16=not args.fp32,
        num_workers=args.num_workers,
        print_freq=args.print_freq,
    )
    print(f'mAP ({args.split}): {mAP:.2f}%')


if __name__ == '__main__':
    main()
