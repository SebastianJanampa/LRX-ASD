"""Evaluate an ASD model on the Columbia ASD dataset (per-speaker accuracy and F1).

``--data-path`` is the ``colDataPath`` folder produced by LR-ASD's
``Columbia_test.py --evalCol`` (it contains ``col/{pycrop,pyframes,pywork}`` and
``col_labels/``).

Example:
    python tools/eval_columbia.py --model lrxasd --weights weights/lrx_asd.model \
        --data-path colDataPath
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
    parser.add_argument('--data-path', required=True, help='colDataPath folder')
    parser.add_argument('--durations', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='chunk lengths in seconds; their scores are averaged with equal weights')
    parser.add_argument('--duration-detailed', action='store_true', help='also report each duration separately')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    parser.add_argument('--fp16', action='store_true', help='float16 inference')
    return parser.parse_args()


def main():
    """Load the model and print the Columbia metrics."""
    args = parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = Model(args.model, device=args.device)
    model.load_weights(args.weights)
    model(
        dataPath=os.path.join(args.data_path, 'col', 'pycrop'),
        durationSet={d: 1 for d in args.durations},
        use_fp16=args.fp16,
        duration_detailed=args.duration_detailed,
        evaluate=True,
        gtPath=os.path.join(args.data_path, 'col_labels'),
        isColumbia=True,
    )


if __name__ == '__main__':
    main()
