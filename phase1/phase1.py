"""
Python 3.12.6

To install dependencies run either:
pip install librosa matplotlib numpy soundfile ipython
or
pip install -r requirements.txt (requires python 3.12.6 for compatibility)

To run:
python phase1.py
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio


def read_and_resample(file_path):
    # Read the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Check if stereo or mono
    channels = "Stereo" if y.ndim > 1 and y.shape[0] == 2 else "Mono"

    print(f"File: {file_path}")
    print(f"Channels: {channels}")
    print(f"Original sampling rate: {sr} Hz")

    # Resample to 16 kHz if necessary
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16 kHz")

    return y, sr


def play_audio(y, sr):
    return Audio(y, rate=sr)


def save_audio(y, sr, file_path):
    sf.write("processed_" + file_path, y, sr)
    print(f"Saved audio to {file_path}")


# plot_waveform generates and saves the waveform of the file as a function of its sample number
def plot_waveform(y, sr, title):
    sample_numbers = np.arange(len(y))
    plt.figure(figsize=(12, 6))
    plt.plot(sample_numbers, y)
    plt.title(f"{title} waveform as a function of sample number")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()


def generate_cosine(freq, y, sr):
    t = np.arange(len(y)) / sr
    cosine_signal = np.cos(2 * np.pi * freq * t)
    return cosine_signal


def plot_cosine(signal, freq, sr, title):
    period = 1 / freq
    t = np.arange(len(signal)) / sr
    plt.figure(figsize=(12, 6))
    plt.plot(t[: int(2 * period * sr)], signal[: int(2 * period * sr)])
    plt.title("Two Cycles of 1 kHz Cosine Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    files = ["fox_white_noise.wav"]  # Add other sound files here
    FREQ = 1500
    for fp in files:
        y, sr = read_and_resample(fp)
        cosine_signal = generate_cosine(FREQ, y, sr)

        plot_waveform(y, sr, "waveform_" + fp)
        plot_cosine(cosine_signal, FREQ, sr, "cosine" + fp)

        save_audio(cosine_signal, sr, "cosine_signal_" + fp)
        save_audio(y, sr, "16khz_" + fp)
