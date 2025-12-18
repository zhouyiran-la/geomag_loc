import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


DEFAULT_DATA_NAME = "0_data_with_label_ghw\u5300\u901f1_normalize_W_300_S_100.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FFT and STFT for magnetic sequences.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/plot_data") / DEFAULT_DATA_NAME,
        help="Path to the npz file that contains X_mag.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot_outputs/magnetic_frequency"),
        help="Directory where PNG files will be written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="Sampling rate (Hz) used for FFT frequency axis.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=64,
        help="FFT size for STFT.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=16,
        help="Hop length for STFT.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of sequences to plot (process all when None).",
    )
    parser.add_argument(
        "--axis-labels",
        type=str,
        default="m_x,m_y,m_z",
        help="Comma separated labels for magnetic axes.",
    )
    return parser.parse_args()


def ensure_dirs(base_dir: Path):
    (base_dir / "fft").mkdir(parents=True, exist_ok=True)
    (base_dir / "stft").mkdir(parents=True, exist_ok=True)


def plot_fft(sample_idx: int, seq: np.ndarray, axis_labels, sample_rate: float, out_dir: Path):
    seq = seq - seq.mean(axis=0, keepdims=True)
    freq = np.fft.rfftfreq(seq.shape[0], d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(seq, axis=0))

    fig, axes = plt.subplots(len(axis_labels), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        ax.plot(freq, spectrum[:, dim])
        ax.set_ylabel(f"|FFT| ({axis_labels[dim]})")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Sample {sample_idx} FFT magnitude")
    fig.tight_layout()
    fig.savefig(out_dir / "fft" / f"sample_{sample_idx:03d}_fft.png", dpi=200)
    plt.close(fig)


def plot_stft(sample_idx: int, seq: np.ndarray, axis_labels, n_fft: int, hop_length: int, out_dir: Path):
    tensor = torch.from_numpy(seq).float().transpose(0, 1)
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)

    fig, axes = plt.subplots(1, len(axis_labels), figsize=(15, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for dim, ax in enumerate(axes):
        spec = transform(tensor[dim])
        spec = torch.log1p(spec).numpy()
        im = ax.imshow(
            spec,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_ylabel(axis_labels[dim])
        ax.set_xlabel("Time frame")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle(f"Sample {sample_idx} STFT (log magnitude)")
    fig.tight_layout()
    fig.savefig(out_dir / "stft" / f"sample_{sample_idx:03d}_stft.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dirs(args.output_dir)

    data = np.load(args.data_path)
    if "X_mag" not in data:
        raise ValueError(f"{args.data_path} does not contain key 'X_mag'.")
    x_mag = data["X_mag"]

    axis_labels = [label.strip() for label in args.axis_labels.split(",") if label.strip()]
    if len(axis_labels) != x_mag.shape[-1]:
        axis_labels = [f"dim_{i}" for i in range(x_mag.shape[-1])]

    num_samples = x_mag.shape[0] if args.max_samples is None else min(args.max_samples, x_mag.shape[0])
    for idx in range(num_samples):
        seq = x_mag[idx]
        plot_fft(idx, seq, axis_labels, args.sample_rate, args.output_dir)
        plot_stft(idx, seq, axis_labels, args.n_fft, args.hop_length, args.output_dir)

    print(f"Saved FFT and STFT plots for {num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
