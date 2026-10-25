"""Run active speaker detection on pre-processed face tracks.

The input is the ``pycrop`` folder made by the LR-ASD / TalkNet demo pipeline
(``Columbia_test.py --videoName <name> --videoFolder <folder>``): one
``<track>.avi`` and one ``<track>.wav`` per face track. Per-frame speaking
scores are saved as a pickled list (one array per track) to
``<save-path>`` (default: ``<pywork>/scores.pckl``), the format that LR-ASD's
``visualization()`` reads.

Example:
    python tools/infer.py --model lrxasd --weights weights/lrx_asd.model \
        --crop-dir demo/my_video/pycrop
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyasd.engine.model import MODELS, Model  # noqa: E402


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lrxasd', choices=list(MODELS), help='architecture to use')
    parser.add_argument('--weights', required=True, help='model weights (.model) or training checkpoint (.pth)')
    parser.add_argument('--crop-dir', required=True, help='pycrop folder with <track>.avi / <track>.wav pairs')
    parser.add_argument('--durations', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='chunk lengths in seconds; their scores are averaged with equal weights')
    parser.add_argument('--save-path', default=None,
                        help='output pickle (default: <crop-dir>/../pywork/scores.pckl)')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    parser.add_argument('--fp16', action='store_true', help='float16 inference')
    return parser.parse_args()


def main():
    """Score every face track and save the results."""
    args = parse_args()

    crop_dir = os.path.normpath(args.crop_dir)
    save_path = args.save_path or os.path.join(os.path.dirname(crop_dir), 'pywork', 'scores.pckl')

    model = Model(args.model, device=args.device)
    model.load_weights(args.weights)
    scores = model(
        dataPath=crop_dir,
        durationSet={d: 1 for d in args.durations},
        use_fp16=args.fp16,
    )

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scores, f)

    for i, s in enumerate(scores):
        print(f'track {i:05d}: {len(s)} frames, speaking {100 * (s > 0.5).mean():.1f}% of the time')
    print(f'Saved scores of {len(scores)} tracks to {save_path}')


if __name__ == '__main__':
    main()
