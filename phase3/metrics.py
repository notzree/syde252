import librosa
import numpy as np
import time

def calculate_snr(original_signal, processed_signal):
    noise = original_signal - processed_signal
    signal_power = np.mean(original_signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_mfcc_similarity(original_signal, processed_signal, sr):
    mfcc_original = librosa.feature.mfcc(y=original_signal, sr=sr)
    mfcc_processed = librosa.feature.mfcc(y=processed_signal, sr=sr)
    mse = np.mean((mfcc_original - mfcc_processed)**2)
    return mse

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

def evaluate_metrics(original_signal, processed_signal, sr):
    snr = calculate_snr(original_signal, processed_signal)
    mfcc_similarity = calculate_mfcc_similarity(original_signal, processed_signal, sr)
    freq_representation, processing_time = measure_processing_speed(
        assess_frequency_representation, original_signal, processed_signal, sr
    )

    return {
        "snr": snr,
        "mfcc_similarity": mfcc_similarity,
        "freq_representation": freq_representation,
        "processing_time": processing_time
    }
