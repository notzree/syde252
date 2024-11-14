import librosa
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os


# Phase 1 functions
def read_and_resample(file_path):
    y, sr = librosa.load(file_path, sr=None)
    channels = "Stereo" if y.ndim > 1 and y.shape[0] == 2 else "Mono"
    print(f"File: {file_path}")
    print(f"Channels: {channels}")
    print(f"Original sampling rate: {sr} Hz")
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16 kHz")
    return y, sr


def save_audio(y, sr, file_path):
    # Extract the base name to avoid directory paths in the output
    base_name = os.path.basename(file_path)
    output_path = "processed_" + base_name

    sf.write(output_path, y, sr)
    print(f"Saved audio to {output_path}")


# Phase 2 functions
def create_bandpass_filters(num_bands, fs):
    # Define frequency range from 100 Hz to slightly below the Nyquist limit (Nyquist = fs / 2)
    nyquist = fs / 2
    max_freq = 0.99 * nyquist  # Ensure the highest frequency is below the Nyquist limit
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


# Plotting functions
def plot_signal(y, sr, title, xlabel="Sample Number", ylabel="Amplitude"):
    # Ensure the directory for saving the plot exists
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


def combine_bands(band_signals):
    """
    Combine multiple frequency bands back into a single signal

    Parameters:
    band_signals: List of numpy arrays containing the band-limited signals

    Returns:
    Combined audio signal
    """
    return np.sum(band_signals, axis=0)


if __name__ == "__main__":
    # Configuration
    files = ["../data/fox_white_noise.wav"]  # Add your audio file paths here
    num_bands = 8
    cutoff_freq = 400  # Hz

    for fp in files:
        # Phase 1: Read and resample audio
        y, sr = read_and_resample(fp)

        # Phase 2: Bandpass filter bank creation
        filters = create_bandpass_filters(num_bands, sr)

        # Apply bandpass filters to the audio signal
        filtered_signals = apply_filters(y, filters)

        save_band_signals(filtered_signals, sr, fp, "bandpass_output")

        # Combine bands and save (to verify they are split correctly)
        combined_band = combine_bands(filtered_signals)
        save_band_signals([combined_band], sr, fp, "output", "bandpass_combined.wav")

        # Plot the lowest and highest frequency channel outputs
        plot_signal(filtered_signals[0], sr, f"output/Lowest frequency channel output_{fp}")
        plot_signal(filtered_signals[-1], sr, f"output/Highest frequency channel output_{fp}")

        # Rectify the filtered signals
        rectified_signals = rectify_signals(filtered_signals)

        # Create and apply a lowpass filter for envelope extraction
        b_lpf, a_lpf = create_lowpass_filter(cutoff_freq, sr)
        envelope_signals = apply_lowpass_filter(rectified_signals, b_lpf, a_lpf)

        # Plot the extracted envelope of the lowest and highest frequency channels
        plot_signal(envelope_signals[0], sr, f"output/Extracted envelope of the lowest frequency channel_{fp}")
        plot_signal(envelope_signals[-1], sr, f"output/Extracted envelope of the highest frequency channel_{fp}")

        # Save the processed audio outputs (optional)
        save_audio(envelope_signals[0], sr, f"output/envelope_lowest_channel_{fp}")
        save_audio(envelope_signals[-1], sr, f"output/envelope_highest_channel_{fp}")
