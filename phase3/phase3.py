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
from typing import List, Tuple
from numpy.typing import NDArray
from scipy.io import wavfile
import os
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import asdict
from metrics import evaluate_metrics
from dataclasses import dataclass
import heapq
import time


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


class EnvelopeDetector(System):
    def __init__(
        self,
        sr: int,
        cutoff_freq: float = 150.0,
        filter_order: int = 4,
        detector_type: str = "peak",
        smoothing_factor: float = 0.3,
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


@dataclass
class FrequencyGeneratorReport:
    spacing: str
    overlap: int


class FrequencyRangeGenerator(ABC):
    @abstractmethod
    def get_freq_tuple(self, i: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_nyquist(self) -> float:
        pass

    @abstractmethod
    def report(self) -> FrequencyGeneratorReport:
        pass


class LinearRangeGen(FrequencyRangeGenerator):
    def __init__(self, sr, num_bands):
        self.nyquist = sr / 2
        max_freq = 0.99 * self.nyquist
        self.freq_range = np.linspace(100, max_freq, num_bands + 1)

    def get_freq_tuple(self, i: int) -> Tuple[float, float]:
        return (self.freq_range[i] / self.nyquist, self.freq_range[i + 1] / self.nyquist)

    def get_nyquist(self):
        return self.nyquist

    def report(self):
        return FrequencyGeneratorReport(spacing="linear", overlap=0)


class LogRangeGen(FrequencyRangeGenerator):
    def __init__(self, sr, num_bands):
        self.nyquist = sr / 2
        max_freq = 0.99 * self.nyquist
        self.freq_range = np.logspace(np.log10(100), np.log10(max_freq), num_bands + 1)

    def get_freq_tuple(self, i: int) -> Tuple[float, float]:
        return (self.freq_range[i] / self.nyquist, self.freq_range[i + 1] / self.nyquist)

    def get_nyquist(self):
        return self.nyquist

    def report(self):
        return FrequencyGeneratorReport(spacing="logarithmic", overlap=0)


class OverlappingRangeGen(FrequencyRangeGenerator):
    def __init__(self, sr: float, num_bands: int, overlap_percent: float, log: bool):
        self.nyquist = sr / 2
        self.overlap_percent = overlap_percent
        self.log = log
        max_freq = 0.99 * self.nyquist
        min_freq = 100
        if not 0 < overlap_percent <= 100:
            raise ValueError("overlap_percent must be between 1 and 100")
        # Convert percentage to fraction
        overlap_fraction = overlap_percent / 100

        # Calculate how many additional bands we need to account for overlap
        # If bands overlap by 50%, we need 2x as many points to maintain num_bands
        effective_points = num_bands / (1 - overlap_fraction)
        if log:
            all_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), int(np.ceil(effective_points)))
        else:
            all_freqs = all_freqs = np.linspace(min_freq, max_freq, int(np.ceil(effective_points)))

        # Calculate the number of points to shift for each band
        points_per_band = len(all_freqs) // num_bands

        # Initialize arrays for lower and upper cutoff frequencies
        lower_cutoffs = np.zeros(num_bands)
        upper_cutoffs = np.zeros(num_bands)

        for i in range(num_bands):
            start_idx = int(i * points_per_band * (1 - overlap_fraction))
            end_idx = start_idx + points_per_band

            if end_idx > len(all_freqs):
                end_idx = len(all_freqs)

            lower_cutoffs[i] = all_freqs[start_idx] / self.nyquist
            upper_cutoffs[i] = all_freqs[min(end_idx, len(all_freqs) - 1)] / self.nyquist
        self.lower_cutoffs = lower_cutoffs
        self.upper_cutoffs = upper_cutoffs

    def get_freq_tuple(self, i):
        return (self.lower_cutoffs[i], self.upper_cutoffs[i])

    def get_nyquist(self):
        return self.nyquist

    def report(self):
        spc = "logarithmic" if self.log else "linear"
        return FrequencyGeneratorReport(spacing=spc, overlap=self.overlap_percent)


class CochlearImplant(System):
    def __init__(
        self, num_bands, signal_length, sr, lowpass_cutoff, fg: FrequencyRangeGenerator, sf: float, pd_type: str
    ):
        nyquist = sr / 2
        if nyquist != fg.get_nyquist():
            raise ValueError(
                f"expected frequency generator nyquist ({fg.get_nyquist()}) to match system nyquist ({nyquist})"
            )
        self.num_bands = num_bands
        channels: List[Channel] = []

        for i in range(num_bands):
            low, high = fg.get_freq_tuple(i)
            center_freq = (low * nyquist + high * nyquist) / 2  # un-normalized center-freq
            if 0 < low < 1 and 0 < high < 1:
                chan = Channel(center_freq, generate_cosine(center_freq, signal_length, sr))
                bandpass_filter = Filter(butter(N=4, Wn=[low, high], btype="band"))  # Create Bandpass Filter
                chan.add_system(bandpass_filter)
                ed = EnvelopeDetector(sr, lowpass_cutoff, 6, pd_type, sf)
                chan.add_system(ed)
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


def save_band_signals(signals, sample_rate, input_file, output_dir="output", file_name=""):
    """
    Save each frequency band as a separate WAV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension and directory
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Save each band
    for i, signal in enumerate(signals):
        scaled_signal = np.int16(signal * 32767)

        name = f"{base_name}_band_{i}.wav" if not file_name else file_name
        output_file = os.path.join(output_dir, name)

        wavfile.write(output_file, sample_rate, scaled_signal)
        print(f"Saved band {i} to {output_file}")


@dataclass
class TestingParams:
    overall_score: float
    num_bands: int
    cutoff_freq: int
    overlap: int
    spacing: str
    peak_detection_type: str
    pesq_score: float
    snr_score: float
    processing_time: float
    output_signal: List[float]

    def __str__(self):
        return (
            f"overall_score: {self.num_bands:.3f}\n"
            f"number of bands: {self.num_bands:.3f}\n"
            f"cutoff frequency: {self.cutoff_freq:.3f} hz\n"
            f"overlap: {self.overlap}\n"
            f"spacing: {self.spacing}\n"
            f"PESQ Score: {self.pesq_score:.3f}\n"
            f"SNR Score: {self.snr_score:.3f} dB\n"
            f"Processing Time: {self.processing_time:.3f} s"
        )

    def __lt__(self, other: "TestingParams"):
        return self.pesq_score < other.pesq_score


# helper function to return a list of all of our frequency generators
def enumerate_frequency_generators(
    low: int,
    high: int,
    step: int,
    sr: int,
    num_bands: int,
):
    generators: List[FrequencyRangeGenerator] = [LinearRangeGen(sr, num_bands), LogRangeGen(sr, num_bands)]
    for overlap_percent in range(low, high, step):
        for is_log in [True, False]:
            overlap_generator = OverlappingRangeGen(
                sr=sr, num_bands=num_bands, overlap_percent=overlap_percent, log=is_log
            )
            generators.append(overlap_generator)
    return generators


class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.heap: List[TestingParams] = []

    def add(self, val):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)

    def get_top_k(self):
        return sorted(self.heap, reverse=True)


if __name__ == "__main__":
    output_dir = "output_csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [
        "../data/female_another_speaker.wav",
        "../data/female_cafe.wav",
        "../data/female_quiet.wav",
        "../data/female_reverb.wav",
        "../data/female_white_noise.wav",
        "../data/male_another_speaker.wav",
        "../data/male_cafe.wav",
        "../data/male_quiet.wav",
        "../data/male_reverb.wav",
        "../data/male_white_noise.wav",
    ]
    LOW_BAND = 8
    HIGH_BAND = 30
    LOW_CUTOFF = 100
    HIGH_CUTOFF = 700
    LOW_OVERLAP = 6
    HIGH_OVERLAP = 12
    tracker: dict[str, TopKHeap] = {}
    saved_signals: List[dict] = []
    for fp in files:
        print(f"starting analysis on {fp}")
        metrics_list: List[dict] = []
        max_mean_score = 0.0
        best_cutoff, best_bands = 0, 0
        input_signal, sr = read_and_resample(fp)
        tracker[fp] = TopKHeap(3)

        for num_bands in range(LOW_BAND, HIGH_BAND, 2):  # 11
            for cutoff_freq in range(LOW_CUTOFF, HIGH_CUTOFF, 100):  # 6
                for fg in enumerate_frequency_generators(
                    LOW_OVERLAP, HIGH_OVERLAP, 1, sr, num_bands
                ):  # 2 + 2 * (6) = 14
                    for peak_detector_type in ["peak", "rms"]:  # 2
                        # TOtal 11 * 6 * 14 * 2 = 1848 configurations / file
                        implant = CochlearImplant(
                            num_bands=num_bands,
                            signal_length=len(input_signal),
                            sr=sr,
                            lowpass_cutoff=cutoff_freq,
                            fg=fg,
                            sf=0.3,
                            pd_type=peak_detector_type,
                        )
                        start_time = time.time()
                        output_signal = implant.output(input_signal)
                        end_time = time.time()
                        processing_time = end_time - start_time
                        metrics = evaluate_metrics(input_signal, output_signal, sr)
                        metrics_dict = asdict(metrics)
                        metrics_dict["processing_time"] = processing_time
                        metrics_dict["num_bands"] = num_bands
                        metrics_dict["cutoff_freq"] = cutoff_freq
                        fg_report = fg.report()
                        metrics_dict["overlap"] = fg_report.overlap
                        metrics_dict["spacing"] = fg_report.spacing
                        metrics_dict["peak_detector_type"] = peak_detector_type
                        metrics_list.append(metrics_dict)
                        tracker[fp].add(
                            TestingParams(
                                (
                                    0.6 * metrics.pesq
                                    + 0.15 * metrics.freq_representation
                                    + 0.2 * metrics.snr  # negative snr would result in lower score
                                    + 0.05 * (1 - processing_time)
                                ),
                                num_bands,
                                cutoff_freq,
                                fg_report.overlap,
                                fg_report.spacing,
                                peak_detector_type,
                                metrics.pesq,
                                metrics.snr,
                                processing_time,
                                output_signal,
                            )
                        )
        metrics_df = pd.DataFrame(metrics_list)
        sanitized_fp = fp.replace("/", "_")
        csv_file_path = os.path.join(output_dir, f"{sanitized_fp}_implant_metrics.csv")
        metrics_df.to_csv(csv_file_path, index=False)
        print("Saving top 3 pesq scores...")
        ts = []  # top signals
        for tp in tracker[fp].get_top_k():
            print(tp)
            ts.append(tp.output_signal)
            saved_signal_params = asdict(tp)
            saved_signal_params.pop("output_signal", None)  # don't need to record the signal
            saved_signal_params["file"] = fp
            saved_signals.append(saved_signal_params)

        save_band_signals(ts, sr, fp, f"{sanitized_fp}_top3_output")

    saved_sigals_df = pd.DataFrame(saved_signals)
    saved_sigals_df.to_csv(output_dir, "final_saved_signal_data.csv")
