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
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import asdict
from metrics import evaluate_metrics, Metrics


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


from scipy.signal import butter, lfilter
import numpy as np
from abc import ABC, abstractmethod


class EnvelopeDetector(System):
    def __init__(
        self,
        sr: int,
        cutoff_freq: float = 150.0,
        filter_order: int = 4,
        detector_type: str = "peak",
        smoothing_factor: float = 0.1,
    ):
        """
        Initialize envelope detector with configurable parameters.

        Args:
            sr: Sampling rate in Hz
            cutoff_freq: Cutoff frequency for the low-pass filter in Hz
            filter_order: Order of the Butterworth filter
            detector_type: Type of detection - 'peak' or 'rms'
            smoothing_factor: Smoothing factor for RMS detection (0 to 1)
        """
        self.sr = sr
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.detector_type = detector_type
        self.smoothing_factor = smoothing_factor

        # Initialize the low-pass filter
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(filter_order, normalized_cutoff, btype="low")

    def _peak_detect(self, signal: np.ndarray) -> np.ndarray:
        """Perform peak detection on the signal."""
        return np.abs(signal)

    def _rms_detect(self, signal: np.ndarray) -> np.ndarray:
        """Perform RMS detection on the signal."""
        # Square the signal
        squared = np.square(signal)

        # Apply exponential moving average for smoothing
        smoothed = np.zeros_like(squared)
        smoothed[0] = squared[0]
        for i in range(1, len(squared)):
            smoothed[i] = self.smoothing_factor * squared[i] + (1 - self.smoothing_factor) * smoothed[i - 1]

        return np.sqrt(smoothed)

    def output(self, signal: np.ndarray) -> np.ndarray:
        """
        Process the input signal to detect its envelope.

        Args:
            signal: Input signal array

        Returns:
            Envelope of the input signal
        """
        # Step 1: Detection
        if self.detector_type == "peaks":
            detected = self._peak_detect(signal)
        else:  # RMS detection
            detected = self._rms_detect(signal)

        # Step 2: Filtering
        envelope = lfilter(self.b, self.a, detected)

        return envelope

    def set_cutoff(self, new_cutoff: float) -> None:
        """
        Update the cutoff frequency of the low-pass filter.

        Args:
            new_cutoff: New cutoff frequency in Hz
        """
        nyquist = self.sr / 2
        normalized_cutoff = new_cutoff / nyquist
        self.b, self.a = butter(self.filter_order, normalized_cutoff, btype="low")
        self.cutoff_freq = new_cutoff


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


class CochlearImplant(System):
    def __init__(self, num_bands, signal_length, sr, lowpass_cutoff):
        self.num_bands = num_bands
        self.lowpass_cutoff = lowpass_cutoff
        nyquist = sr / 2
        max_freq = 0.99 * nyquist
        # freq_range = np.linspace(100, max_freq, num_bands + 1)
        freq_range = np.logspace(np.log10(100), np.log10(max_freq), num_bands + 1)
        channels: List[Channel] = []
        low_b, low_a = butter(N=4, Wn=lowpass_cutoff / (sr / 2), btype="low")
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
                # envelope_detector = EnvelopeDetector(
                #     sr=16000,
                #     cutoff_freq=cutoff_freq,  # Adjust based on your signal characteristics
                #     detector_type="peak",  # or 'rms' for RMS detection
                # )
                # chan.add_system(envelope_detector)
                chan.add_system(rectifier)
                chan.add_system(lowpass_filter)
                am = AmplitudeModulator(chan.cosine)
                chan.add_system(am)
                channels.append(chan)
        self.channels = channels

    def output(self, signal):
        processed_signals = [channel.pass_signal(signal) for channel in self.channels]
        combined_signal = np.sum(processed_signals, axis=0)
        max_abs_value = np.max(np.abs(combined_signal))
        normalized_signal = combined_signal if max_abs_value == 0 else combined_signal / max_abs_value
        return normalized_signal

    def add_metrics(self, metrics: Metrics) -> None:
        self.metrics = metrics


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
    output_dir = "output_csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metrics_list: List[dict] = []
    files = ["../data/fox_white_noise.wav"]
    LOW_BAND = 2
    HIGH_BAND = 24
    LOW_CUTOFF = 10
    HIGH_CUTOFF = 200
    # num_bands = 8
    # cutoff_freq = 400  # hz
    max_mean_score = 0
    best_cutoff, best_bands = 0, 0
    for fp in files:
        input_signal, sr = read_and_resample(fp)
        implant = CochlearImplant(num_bands=20, signal_length=len(input_signal), sr=sr, lowpass_cutoff=90)
        save_band_signals([implant.output(input_signal)], sr, fp, "best_output_combined")
        # for num_bands in range(LOW_BAND, HIGH_BAND, 2):
        #     for cutoff_freq in range(LOW_CUTOFF, HIGH_CUTOFF, 10):
        #         implant = CochlearImplant(
        #             num_bands=num_bands, signal_length=len(input_signal), sr=sr, lowpass_cutoff=cutoff_freq
        #         )
        #         output_signal = implant.output(input_signal)
        #         output_signal = implant.output(input_signal)
        #         metrics = evaluate_metrics(input_signal, output_signal, sr)
        #         implant.add_metrics(metrics)
        #         metrics_dict = asdict(implant.metrics)

        #         # score_sum = 0
        #         # for key, value in metrics_dict.items():
        #         #     if key == "processing_time" or key == "mfcc_similarity":
        #         #         continue
        #         #     score_sum += value
        #         # score_sum // 4
        #         score_sum = metrics.pesq
        #         if score_sum > max_mean_score:
        #             best_bands, best_cutoff = num_bands, cutoff_freq
        #             max_mean_score = score_sum
        #         metrics_dict["num_bands"] = num_bands
        #         metrics_dict["cutoff_freq"] = cutoff_freq
        #         metrics_list.append(metrics_dict)
        # metrics_df = pd.DataFrame(metrics_list)
        # csv_file_path = os.path.join(output_dir, "implant_metrics.csv")
        # metrics_df.to_csv(csv_file_path, index=False)
        # print("BEST PESQ SCORE COMBINATION")
        # print(max_mean_score, best_bands, best_cutoff)
