"""
Pure NumPy + soundfile mel-spectrogram feature pipeline. No librosa.

Full Spectrum's recipe step 2 names this explicitly: "compute features in
plain NumPy and soundfile ... SiloSense avoided librosa for exactly this
reason: no dependency chain that fails to compile on-device." This module
is the audio-side equivalent of SiloSense's own feature pipeline, and the
same code trains the model and (eventually) runs on-device -- the numeric
identity that recipe step 8's diff-test checks.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000        # MIMII's native rate
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64                  # already a power of 2: clean for 3x stride-2 conv downsampling
FMIN = 50.0
FMAX = SAMPLE_RATE / 2

#: Fixed at 256 frames (also power-of-2-friendly) so a conv autoencoder's
#: three stride-2 downsampling steps land on exact integer shapes with no
#: post-hoc crop/pad inside the model. Solved for CLIP_SAMPLES, not the
#: other way around: n_frames = 1 + (CLIP_SAMPLES - N_FFT) // HOP_LENGTH.
#: MIMII's own clips run ~10s (160,000 samples); this truncates to the
#: leading ~8.2s of each, never pads a real file.
TARGET_FRAMES = 256
CLIP_SAMPLES = (TARGET_FRAMES - 1) * HOP_LENGTH + N_FFT
CLIP_SECONDS = CLIP_SAMPLES / SAMPLE_RATE


def load_mono(path: str) -> np.ndarray:
    """
    Read a wav file and mix down to mono.

    MIMII's files are 8-channel (microphone array). Averaging channels is a
    deliberate simplification for this bounded demo, not a claim that
    per-channel information is worthless -- a real deployment would likely
    keep the channel closest to the fault source. Fixed-length: padded with
    zeros or truncated to CLIP_SAMPLES so every feature matrix that comes
    out the other end has the same shape, which a fixed-input-size model
    needs.
    """
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE}Hz, got {sr}Hz")
    mono = samples.mean(axis=1)
    if len(mono) < CLIP_SAMPLES:
        mono = np.pad(mono, (0, CLIP_SAMPLES - len(mono)))
    else:
        mono = mono[:CLIP_SAMPLES]
    return mono


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_magnitude(signal: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Plain-NumPy STFT magnitude. Shape: (n_fft//2 + 1, n_frames)."""
    window = _hann(n_fft)
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, n_fft),
        strides=(signal.strides[0] * hop_length, signal.strides[0]),
    )
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.abs(spectrum).T  # (freq_bins, n_frames)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
    fmin: float = FMIN, fmax: float = FMAX,
) -> np.ndarray:
    """Standard triangular mel filterbank. Shape: (n_mels, n_fft//2 + 1)."""
    n_freq = n_fft // 2 + 1
    mel_min, mel_max = _hz_to_mel(np.array([fmin, fmax]))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb


_FB = mel_filterbank()


def log_mel_spectrogram(signal: np.ndarray) -> np.ndarray:
    """(n_mels, n_frames) log-mel spectrogram, the model's actual input."""
    mag = stft_magnitude(signal)
    mel = _FB @ mag
    return np.log(mel + 1e-6).astype(np.float32)


def extract_features(path: str) -> np.ndarray:
    """One wav file -> one fixed-shape (N_MELS, n_frames) feature matrix."""
    return log_mel_spectrogram(load_mono(path))


def feature_shape() -> tuple[int, int]:
    """The fixed (N_MELS, n_frames) shape every extract_features() call produces."""
    n_frames = 1 + (CLIP_SAMPLES - N_FFT) // HOP_LENGTH
    return (N_MELS, n_frames)
