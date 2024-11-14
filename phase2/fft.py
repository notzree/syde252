import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os


def normalize_audio(audio_data):
    """
    Normalize audio data to float between -1 and 1, handling both int and float inputs

    Parameters:
    audio_data: numpy array of audio samples

    Returns:
    Normalized audio data as float
    """
    # Convert to float if not already
    audio_float = audio_data.astype(float)

    # If input was integer type, normalize by max possible value
    if np.issubdtype(audio_data.dtype, np.integer):
        audio_float /= np.iinfo(audio_data.dtype).max
    # If input was float type, normalize by max absolute value if needed
    elif np.abs(audio_float).max() > 1.0:
        audio_float /= np.abs(audio_float).max()

    return audio_float


def split_into_frequency_bands(audio_signal, sample_rate, num_bands):
    """
    Split an audio signal into N frequency bands using FFT.
    """
    # Compute FFT
    fft_result = fft(audio_signal)
    frequencies = np.fft.fftfreq(len(audio_signal), 1 / sample_rate)

    # We only need the positive frequencies (up to Nyquist frequency)
    positive_freq_indices = np.where(frequencies >= 0)[0]
    max_freq = frequencies[positive_freq_indices].max()

    # Calculate frequency bands (using logarithmic spacing for better perceptual results)
    min_freq = 20  # Starting from 20 Hz
    bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)

    # Initialize storage for each band
    band_signals = []
    band_ranges = {}

    # Create mask for each frequency band and inverse FFT
    for i in range(num_bands):
        mask = np.zeros_like(fft_result)

        band_start = bands[i]
        band_end = bands[i + 1]

        band_ranges[i] = (band_start, band_end)

        # Create the mask for both positive and negative frequencies
        mask_positive = (frequencies >= band_start) & (frequencies < band_end)
        mask_negative = (frequencies <= -band_start) & (frequencies > -band_end)
        mask[mask_positive | mask_negative] = 1

        # Apply mask and inverse FFT
        band_fft = fft_result * mask
        band_signal = np.real(ifft(band_fft))

        band_signals.append(band_signal)

    return band_signals, band_ranges


def process_audio_file(input_file, num_bands):
    """
    Process an audio file by splitting it into frequency bands
    """
    # Read audio file
    sample_rate, audio_data = wavfile.read(input_file)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio data using the new function
    audio_data = normalize_audio(audio_data)

    # Split into bands
    band_signals, band_ranges = split_into_frequency_bands(audio_data, sample_rate, num_bands)

    return band_signals, band_ranges, sample_rate


def combine_bands(band_signals):
    """
    Combine multiple frequency bands back into a single signal

    Parameters:
    band_signals: List of numpy arrays containing the band-limited signals

    Returns:
    Combined audio signal
    """
    return np.sum(band_signals, axis=0)


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


# Process your file
input_file = "../data/fox_white_noise.wav"
signals, ranges, sample_rate = process_audio_file(input_file, 6)

# Save all bands
save_band_signals(signals, sample_rate, input_file)

# Print frequency ranges for each band
for band_num, (low_freq, high_freq) in ranges.items():
    print(f"Band {band_num}: {low_freq:.1f} Hz - {high_freq:.1f} Hz")

# Combine bands and save (to verify they are split correctly)
combined_band = combine_bands(signals)
save_band_signals([combined_band], sample_rate, input_file, "output", "combined_bands.wav")
