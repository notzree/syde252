"""
Python 3.12.6

To install dependencies run either:
pip install librosa matplotlib numpy soundfile scipy
or
pip install -r requirements.txt (requires python 3.12.6 for compatibility)

To run:
python phase3.py
"""

import librosa
import numpy as np
from scipy.signal import butter, lfilter
from typing import List, Any
from numpy.typing import NDArray
from scipy.io import wavfile
import os
from abc import ABC, abstractmethod


class System(ABC):
    @abstractmethod
    def output(self, signal):
        """Process the input signal and return an output signal."""
        pass


class Filter(System):
    def __init__(self, filter):
        self.filter = filter

    def output(self, signal):
        # filters
        b, a = self.filter
        return lfilter(b, a, signal)


class Rectifier(System):
    def __init__(self):
        pass

    def output(self, signal):
        return np.abs(signal)


class AmplitudeModulator(System):
    def __init__(self, original_signal):
        self.original_signal = original_signal

    def output(self, incoming_signal):
        return incoming_signal * self.original_signal


class Channel:
    def __init__(self, cf: int, cosine):
        self.cf = cf
        self.cosine = cosine
        self.systems: List[System] = []

    def add_system(self, system: System):
        self.systems.append(system)

    def pass_signal(self, input_signal):
        for system in self.systems:
            input_signal = system.output(input_signal)
        return input_signal


# task 10: Generate a Signal Using a Cosine Function for Each Channel
def generate_cosine(freq, signal_length, sr) -> NDArray[np.float64]:
    t = np.arange(signal_length) / sr
    cosine_signal = np.cos(2 * np.pi * freq * t)
    return cosine_signal


def create_channels(num_bands, signal_length, sr) -> List[Channel]:
    nyquist = sr / 2
    max_freq = 0.99 * nyquist
    freq_range = np.linspace(100, max_freq, num_bands + 1)
    channels: List[Channel] = []
    low_b, low_a = butter(N=4, Wn=2000 / (sr / 2), btype="low")
    for i in range(num_bands):
        low = freq_range[i] / nyquist
        high = freq_range[i + 1] / nyquist
        if 0 < low < 1 and 0 < high < 1:
            b, a = butter(N=4, Wn=[low, high], btype="band")
            bandpass_filter = Filter((b, a))  # Create Bandpass Filter
            lowpass_filter = Filter((low_b, low_a))
            rectifier = Rectifier()
            center_freq = (freq_range[i] + freq_range[i + 1]) / 2
            chan = Channel(center_freq, generate_cosine(center_freq, signal_length, sr))
            chan.add_system(bandpass_filter)
            chan.add_system(rectifier)
            chan.add_system(lowpass_filter)
            am = AmplitudeModulator(chan.cosine)
            chan.add_system(am)
            channels.append(chan)
        else:
            raise ValueError(f"Critical frequencies must be in range (0, 1). Got low: {low}, high: {high}")
    return channels


def read_and_resample(file_path):
    # Extract file path and sampling rate of sound file
    y, sr = librosa.load(file_path, sr=None)
    channels = "Stereo" if y.ndim > 1 and y.shape[0] == 2 else "Mono"
    print(f"File: {file_path}")
    print(f"Channels: {channels}")
    print(f"Original sampling rate: {sr} Hz")
    # Resample to 16kHz if original sampling rate is not 16kHz
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16 kHz")
    return y, sr


def save_band_signals(signals, sample_rate, input_file, output_dir="output", input_name=""):
    """
    Save each frequency band as a separate WAV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension and directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Save each band
    for i, signal in enumerate(signals):
        # Scale the signal to 16-bit integer range
        scaled_signal = np.int16(signal * 32767)

        # Generate output filename
        name = f"{base_name}_band_{i}.wav" if not input_name else input_name
        output_file = os.path.join(output_dir, name)

        # Save the file
        wavfile.write(output_file, sample_rate, scaled_signal)
        print(f"Saved band {i} to {output_file}")


if __name__ == "__main__":
    files = ["../data/fox_white_noise.wav"]  # Add other audio files here
    num_bands = 8
    cutoff_freq = 400  # hz
    for fp in files:
        y, sr = read_and_resample(fp)
        channels = create_channels(num_bands=num_bands, signal_length=len(y), sr=sr)

        # Process the input signal through each channel
        processed_signals = [channel.pass_signal(y) for channel in channels]
        save_band_signals(processed_signals, sr, fp, "phase3_output")
        # Add all processed signals together
        combined_signal = np.sum(processed_signals, axis=0)

        max_abs_value = np.max(np.abs(combined_signal))
        if max_abs_value == 0:
            normalized_signal = combined_signal  # Prevent division by zero
        normalized_signal = combined_signal / max_abs_value
        save_band_signals([normalized_signal], sr, fp, "phase3_combined_output_2khz")

        # You can now use the combined_signal for further processing or output
        print(f"Combined and normalized signal shape: {normalized_signal.shape}")
