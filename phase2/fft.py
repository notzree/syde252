import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd


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

    return band_signals, band_ranges, sample_rate, audio_data


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


def compare_signals(original_signal, combined_signal, sample_rate=None):
    """
    Compare two audio signals using multiple metrics to assess their similarity.

    Parameters:
    original_signal: numpy array of the original audio signal
    combined_signal: numpy array of the reconstructed/combined signal
    sample_rate: optional, sampling rate of the signals for frequency domain analysis

    Returns:
    Dictionary containing various similarity metrics
    """
    import numpy as np
    from scipy.stats import pearsonr

    # Ensure signals are the same length
    min_length = min(len(original_signal), len(combined_signal))
    original = original_signal[:min_length]
    combined = combined_signal[:min_length]

    # 1. Mean Square Error (MSE)
    mse = np.mean((original - combined) ** 2)

    # 2. Root Mean Square Error (RMSE)
    rmse = np.sqrt(mse)

    # 3. Peak Signal-to-Noise Ratio (PSNR)
    max_possible = max(np.max(np.abs(original)), np.max(np.abs(combined)))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(max_possible) - 10 * np.log10(mse)

    # 4. Correlation Coefficient
    correlation, _ = pearsonr(original, combined)

    # 5. Maximum Absolute Difference
    max_diff = np.max(np.abs(original - combined))

    # 6. Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - combined) ** 2)
    if noise_power == 0:
        snr = float("inf")
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    # 7. Normalized Root Mean Square Error (NRMSE)
    nrmse = rmse / (np.max(original) - np.min(original))

    # Optional frequency domain comparison if sample_rate is provided
    spectral_difference = None
    if sample_rate is not None:
        # Compute FFTs
        original_fft = np.abs(np.fft.fft(original))
        combined_fft = np.abs(np.fft.fft(combined))

        # Compare spectral content (up to Nyquist frequency)
        nyquist_bin = len(original_fft) // 2
        spectral_difference = np.mean(np.abs(original_fft[:nyquist_bin] - combined_fft[:nyquist_bin])) / np.mean(
            np.abs(original_fft[:nyquist_bin])
        )

    # Compile results
    results = {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "psnr_db": psnr,
        "correlation": correlation,
        "max_difference": max_diff,
        "snr_db": snr,
    }

    if spectral_difference is not None:
        results["spectral_difference"] = spectral_difference

    return results


def print_comparison_results(results):
    """
    Print the comparison results in a formatted way

    Parameters:
    results: Dictionary containing the comparison metrics
    """
    print("\nSignal Comparison Results:")
    print("-" * 50)
    print(f"Mean Square Error (MSE): {results['mse']:.6f}")
    print(f"Root Mean Square Error (RMSE): {results['rmse']:.6f}")
    print(f"Normalized RMSE: {results['nrmse']:.6f}")
    print(f"Peak Signal-to-Noise Ratio: {results['psnr_db']:.2f} dB")
    print(f"Correlation Coefficient: {results['correlation']:.6f}")
    print(f"Maximum Absolute Difference: {results['max_difference']:.6f}")
    print(f"Signal-to-Noise Ratio: {results['snr_db']:.2f} dB")
    if "spectral_difference" in results:
        print(f"Spectral Difference: {results['spectral_difference']:.6f}")


def analyze_band_performance(input_file: str, band_range: range) -> Tuple[Dict, List[Dict]]:
    """
    Analyze audio processing performance with different numbers of frequency bands.

    Parameters:
    input_file: str - Path to the input audio file
    band_range: range - Range of number of bands to test

    Returns:
    Tuple containing best results and all results
    """
    results = []
    best_result = {
        "num_bands": 0,
        "metrics": None,
        "score": float("inf"),  # Using RMSE as primary metric (lower is better)
    }

    for num_bands in band_range:
        print(f"\nProcessing with {num_bands} bands...")

        # Process audio with current number of bands
        signals, ranges, sample_rate, original_signal = process_audio_file(input_file, num_bands)

        # Combine bands
        combined_signal = combine_bands(signals)

        # Compare signals
        comparison = compare_signals(original_signal, combined_signal, sample_rate)

        # Store results
        result = {"num_bands": num_bands, "metrics": comparison, "ranges": ranges}
        results.append(result)

        # Update best result if current RMSE is lower
        if comparison["rmse"] < best_result["score"]:
            best_result = {"num_bands": num_bands, "metrics": comparison, "score": comparison["rmse"]}

        # Print frequency ranges for this configuration
        print(f"\nFrequency ranges for {num_bands} bands:")
        for band_num, (low_freq, high_freq) in ranges.items():
            print(f"Band {band_num}: {low_freq:.1f} Hz - {high_freq:.1f} Hz")

    return best_result, results


def plot_performance_metrics(results: List[Dict]):
    """
    Plot various performance metrics against the number of bands.
    """
    num_bands = [r["num_bands"] for r in results]
    metrics = ["rmse", "correlation", "snr_db", "spectral_difference"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Performance Metrics vs Number of Frequency Bands")

    for ax, metric in zip(axes.flat, metrics):
        values = [r["metrics"][metric] for r in results]
        ax.plot(num_bands, values, "o-")
        ax.set_xlabel("Number of Bands")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)

    plt.tight_layout()
    return fig


# Main execution
def main():
    # Process your file
    input_file = "../data/fox_white_noise.wav"
    # signals, ranges, sample_rate, original_signal = process_audio_file(input_file, 6)

    # # Save all bands
    # save_band_signals(signals, sample_rate, input_file)

    # # Print frequency ranges for each band
    # for band_num, (low_freq, high_freq) in ranges.items():
    #     print(f"Band {band_num}: {low_freq:.1f} Hz - {high_freq:.1f} Hz")

    # # Combine bands and save (to verify they are split correctly)
    # combined_signal = combine_bands(signals)
    # save_band_signals([combined_signal], sample_rate, input_file, "output", "combined_bands.wav")
    # comparaison_results = compare_signals(original_signal, combined_signal, sample_rate)
    # print_comparison_results(comparaison_results)
    band_range = range(2, 21, 2)  # Test even numbers from 2 to 20 bands

    # Run analysis
    best_result, all_results = analyze_band_performance(input_file, band_range)

    # Print best result
    print("\nBest Performance:")
    print(f"Number of bands: {best_result['num_bands']}")
    print("\nMetrics for best configuration:")
    print_comparison_results(best_result["metrics"])

    # Create and save performance plot
    fig = plot_performance_metrics(all_results)
    plt.savefig("band_performance.png")
    plt.close()

    # Prepare data for DataFrame
    df_data = []
    for result in all_results:
        row = {"num_bands": result["num_bands"]}
        row.update(result["metrics"])
        df_data.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(df_data)
    df.to_csv("band_analysis_results.csv", index=False)

    return best_result, all_results


if __name__ == "__main__":
    best_result, all_results = main()
