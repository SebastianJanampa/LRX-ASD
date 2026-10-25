"""PCA visualization of the last visual-encoder layer for several ASD models.

For each model, the first ``--num-tracks`` face tracks of ``--data-path`` go
through the network. The output of ``--layer`` (by default
``visualEncoder.block3``, the last visual encoder block) is projected onto its
3 principal components (shown as RGB) for the frames in ``--frames``. Images are
saved to ``<output-root>/<model_name>_pcas/``.

The source frames and clips are saved once to
``<output-root>/reference_images_pca/``, since every model uses the same tracks.

Example:
    python tools/pca_features.py --data-path colDataPath/col/pycrop
    python tools/pca_features.py --data-path colDataPath/col/pycrop --models lrxasd \
        --weights lrxasd=weights/lrx_asd.pth
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyasd.engine.model import MODELS, Model  # noqa: E402
from pyasd.engine.feature_viz import visualize_feature_maps  # noqa: E402

DEFAULT_WEIGHTS = {
    'lrasd': 'weights/lr_asd.model',
    'lightasd': 'weights/light_asd.model',
    'lrxasd': 'weights/lrx_asd.model',
}


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['lrasd', 'lightasd', 'lrxasd'], choices=list(MODELS),
                        help='models to visualize')
    parser.add_argument('--weights', nargs='*', default=[], metavar='NAME=PATH',
                        help='override the weights of a model, e.g. lrxasd=weights/lrx_asd.pth')
    parser.add_argument('--data-path', required=True,
                        help='pycrop folder with <track>.avi / <track>.wav pairs, e.g. colDataPath/col/pycrop')
    parser.add_argument('--num-tracks', type=int, default=3, help='number of tracks to process (sorted by name)')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 4, 9], help='frame indices to visualize')
    parser.add_argument('--duration', type=float, default=1.0, help='seconds of each track fed to the model')
    parser.add_argument('--layer', default='visualEncoder.block3', help='module whose output is visualized')
    parser.add_argument('--output-root', default='.', help='where the output folders are created')
    parser.add_argument('--device', default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    return parser.parse_args()


def save_reference_images(data_path, num_tracks, frames, duration, out_dir):
    """Save the source frames and clips that the PCA images were computed from.

    The tracks are taken in the same (sorted) order as
    :func:`~pyasd.engine.feature_viz.visualize_feature_maps`. For each track it
    saves:

    - ``<track>_frame{t}.png``: original color face crop (native resolution);
    - ``<track>_frame{t}_input.png``: 112x112 grayscale image fed to the model,
      aligned pixel for pixel with the PCA maps;
    - ``<track>_clip.mp4`` and ``<track>_clip.wav``: the ``duration``-second
      clip, plus ``<track>_clip_with_audio.mp4`` when ffmpeg is available.

    It also writes ``references.json`` with the source files, frame indices and
    timestamps.

    Args:
        data_path (str): ``pycrop`` folder.
        num_tracks (int): Number of tracks.
        frames (list[int]): Frame indices visualized in the PCA images.
        duration (float): Clip length in seconds.
        out_dir (str): Output folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    tracks = sorted(f for f in os.listdir(data_path) if f.endswith('.avi'))[:num_tracks]
    has_ffmpeg = shutil.which('ffmpeg') is not None
    summary = {}

    for avi in tracks:
        name = os.path.splitext(avi)[0]
        avi_path = os.path.join(data_path, avi)
        wav_path = os.path.join(data_path, name + '.wav')

        # Read the frames of the clip
        video = cv2.VideoCapture(avi_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        n_clip = int(round(duration * round(fps)))
        clip = []
        while len(clip) < n_clip:
            ret, frame = video.read()
            if not ret:
                break
            clip.append(frame)
        video.release()

        # Reference frames: color crop and the model input
        saved_frames = []
        for t in frames:
            if t >= len(clip):
                continue
            cv2.imwrite(os.path.join(out_dir, f'{name}_frame{t}.png'), clip[t])
            gray = cv2.resize(cv2.cvtColor(clip[t], cv2.COLOR_BGR2GRAY), (112, 112))
            cv2.imwrite(os.path.join(out_dir, f'{name}_frame{t}_input.png'), gray)
            saved_frames.append(t)

        # Audio segment
        sr, audio = wavfile.read(wav_path)
        clip_wav = os.path.join(out_dir, f'{name}_clip.wav')
        wavfile.write(clip_wav, sr, audio[:int(round(len(clip) / fps * sr))])

        # Video clip (H.264 with ffmpeg, so it plays in slides and browsers)
        clip_mp4 = os.path.join(out_dir, f'{name}_clip.mp4')
        clip_seconds = len(clip) / fps
        if has_ffmpeg:
            base = ['ffmpeg', '-y', '-loglevel', 'error', '-i', avi_path]
            enc = ['-t', f'{clip_seconds:.3f}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
            subprocess.run(base + enc + ['-an', clip_mp4], check=True)
            subprocess.run(base + ['-i', clip_wav] + enc + ['-map', '0:v', '-map', '1:a', '-c:a', 'aac',
                           os.path.join(out_dir, f'{name}_clip_with_audio.mp4')], check=True)
        else:
            h, w = clip[0].shape[:2]
            writer = cv2.VideoWriter(clip_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for frame in clip:
                writer.write(frame)
            writer.release()

        summary[name] = {
            'source_video': os.path.abspath(avi_path),
            'source_audio': os.path.abspath(wav_path),
            'fps': fps,
            'frames': saved_frames,
            'timestamps_sec': [round(t / fps, 3) for t in saved_frames],
            'clip_sec': [0.0, round(clip_seconds, 3)],
        }
        print(f'Saved reference images of track {name}')

    with open(os.path.join(out_dir, 'references.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Compute and save the PCA images for every model, then the reference images."""
    args = parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    for item in args.weights:
        name, path = item.split('=', 1)
        weights[name] = path

    for name in args.models:
        assert name in weights, f'no weights given for {name}, use --weights {name}=<path>'
        print(f'===== {name} ({weights[name]}) =====')
        model = Model(name, device=args.device)
        model.load_weights(weights[name])
        visualize_feature_maps(
            model,
            args.data_path,
            [args.layer],
            num_runs=args.num_tracks,
            duration=args.duration,
            pca_frames=args.frames,
            save=True,
            save_folder=os.path.join(args.output_root, f'{name}_pcas'),
            save_channels=False,
            overwrite=True,
        )

    save_reference_images(args.data_path, args.num_tracks, args.frames, args.duration,
                          os.path.join(args.output_root, 'reference_images_pca'))


if __name__ == '__main__':
    main()
