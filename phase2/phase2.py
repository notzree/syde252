"""
Python 3.12.6

To install dependencies run either:
pip install librosa matplotlib numpy soundfile scipy
or
pip install -r requirements.txt (requires python 3.12.6 for compatibility)

To run:
python phase2.py
"""

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os

# Read sound file and compress to 16kHz
def read_and_resample(file_path):
    # Extract file path and sampling rate of sound file
    y, sr = librosa.load(file_path, sr=None)
    channels = "Stereo" if y.ndim > 1 and y.shape[0] == 2 else "Mono"
    print(f"File: {file_path}")
    print(f"Channels: {channels}")
    print(f"Original sampling rate: {sr} Hz")
    # Resample to 16kHz if original samplign rate is not 16kHz
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16 kHz")
    return y, sr

def save_audio(y, sr, file_path):
    base_name = os.path.basename(file_path)
    output_path = "processed_" + base_name
    sf.write(output_path, y, sr)
    print(f"Saved audio to {output_path}")

# Create 8 bandpass filters from 100Hz to 8kHz
def create_bandpass_filters(num_bands, fs):
    # Define frequency range from 100 Hz to slightly below the Nyquist limit (Nyquist = fs / 2)
    nyquist = fs / 2 # 8kHz
    max_freq = 0.99 * nyquist  # Ensure the highest frequency is below the Nyquist limit (8kHz)
    freq_range = np.linspace(100, max_freq, num_bands + 1)
    filters = []

    for i in range(num_bands):
        # Normalize the frequencies by dividing by the Nyquist frequency
        low = freq_range[i] / nyquist
        high = freq_range[i + 1] / nyquist

        # Ensure the values are within the range (0, 1)
        if 0 < low < 1 and 0 < high < 1:
            b, a = butter(N=4, Wn=[low, high], btype="band")
            filters.append((b, a))
        else:
            raise ValueError(f"Critical frequencies must be in range (0, 1). Got low: {low}, high: {high}")

    return filters


def apply_filters(y, filters):
    filtered_signals = []
    for b, a in filters:
        filtered_signal = lfilter(b, a, y)
        filtered_signals.append(filtered_signal)
    return filtered_signals


def rectify_signals(filtered_signals):
    return [np.abs(signal) for signal in filtered_signals]


def create_lowpass_filter(cutoff_freq, fs):
    b, a = butter(N=4, Wn=cutoff_freq / (fs / 2), btype="low")
    return b, a


def apply_lowpass_filter(signals, b, a):
    return [lfilter(b, a, signal) for signal in signals]


def plot_signal(y, sr, title, xlabel="Sample Number", ylabel="Amplitude"):
    output_dir = os.path.dirname(title)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    files = ["../data/fox_white_noise.wav"]  # Add other audio files here
    num_bands = 8
    cutoff_freq = 400  # hz

    for fp in files:
        y, sr = read_and_resample(fp)
        filters = create_bandpass_filters(num_bands, sr)
        filtered_signals = apply_filters(y, filters)

        # Envelope extraction
        rectified_signals = rectify_signals(filtered_signals)
        b_lpf, a_lpf = create_lowpass_filter(cutoff_freq, sr)
        envelope_signals = apply_lowpass_filter(rectified_signals, b_lpf, a_lpf)

        # Plot envelopes
        plot_signal(envelope_signals[0], sr, f"output/Extracted envelope of the lowest frequency channel_{fp}")
        plot_signal(envelope_signals[-1], sr, f"output/Extracted envelope of the highest frequency channel_{fp}")

        # Plot low/high freq output
        plot_signal(filtered_signals[0], sr, f"output/Lowest frequency channel output_{fp}")
        plot_signal(filtered_signals[-1], sr, f"output/Highest frequency channel output_{fp}")
