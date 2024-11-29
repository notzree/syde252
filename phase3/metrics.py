import librosa
import numpy as np
import time
from dataclasses import dataclass
from scipy.signal import welch
from pesq import pesq


@dataclass
class Metrics:
    snr: float
    spectral_distortion: float
    pesq: float
    mfcc_similarity: float
    freq_representation: float
    processing_time: float

    def __str__(self) -> str:
        return (
            f"Signal-to-Noise Ratio: {self.snr:.2f} dB\n"
            f"Spectral Distortion: {self.spectral_distortion:.2f}\n"
            f"PESQ Score: {self.pesq:.2f}\n"
            f"MFCC Similarity: {self.mfcc_similarity:.2f}\n"
            f"Frequency Representation: {self.freq_representation:.2f}\n"
            f"Processing Time: {self.processing_time:.3f} s"
        )


def calculate_snr(original_signal, processed_signal):
    noise = original_signal - processed_signal
    signal_power = np.mean(original_signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# <--- these 2 functions might be computing similar things --->
def calculate_spectral_distortion(original, processed, fs):
    f, Pxx_orig = welch(original, fs)
    f, Pxx_proc = welch(processed, fs)
    spectral_distortion = np.mean(np.abs(10 * np.log10(Pxx_proc / Pxx_orig)))
    return spectral_distortion


def calculate_mfcc_similarity(original_signal, processed_signal, sr):
    mfcc_original = librosa.feature.mfcc(y=original_signal, sr=sr)
    mfcc_processed = librosa.feature.mfcc(y=processed_signal, sr=sr)
    mse = np.mean((mfcc_original - mfcc_processed) ** 2)
    return mse


# <--->


def calculate_pesq(original, processed, fs):
    """
    PESQ is an objective method for assessing the quality of speech that has been compressed or otherwise processed.
    It provides a score that correlates well with subjective listening tests.
    Importance: PESQ is particularly valuable for evaluating cochlear implant processing because it takes into account human perception of speech quality.
    It can help predict how users might perceive the processed speech, which is crucial for optimizing the implant's parameters for real-world use
    """
    pesq_score = pesq(fs, original, processed, "nb")
    return pesq_score


def assess_frequency_representation(original_signal, processed_signal, sr):
    original_stft = np.abs(librosa.stft(original_signal))
    processed_stft = np.abs(librosa.stft(processed_signal))
    original_mean_freq = np.mean(original_stft, axis=1)
    processed_mean_freq = np.mean(processed_stft, axis=1)
    retention_score = np.sum(np.minimum(processed_mean_freq, original_mean_freq)) / np.sum(original_mean_freq)
    return retention_score


def measure_processing_speed(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    processing_time = end_time - start_time
    return result, processing_time


def evaluate_metrics(original_signal, processed_signal, sr) -> Metrics:
    snr = calculate_snr(original_signal, processed_signal)
    mfcc_similarity = calculate_mfcc_similarity(original_signal, processed_signal, sr)
    freq_representation, processing_time = measure_processing_speed(
        assess_frequency_representation, original_signal, processed_signal, sr
    )
    pesq = calculate_pesq(original_signal, processed_signal, sr)
    spectral_distortion = calculate_spectral_distortion(original_signal, processed_signal, sr)
    return Metrics(
        snr=snr,
        mfcc_similarity=mfcc_similarity,
        pesq=pesq,
        spectral_distortion=spectral_distortion,
        freq_representation=freq_representation,
        processing_time=processing_time,
    )
