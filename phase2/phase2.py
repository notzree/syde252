import librosa
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os
import scipy
from typing import Dict, List, Tuple
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import pearsonr


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
    # freq_range = np.linspace(100, max_freq, num_bands + 1)
    freq_range = np.logspace(np.log10(100), np.log10(max_freq), num_bands + 1)
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


def process_audio_with_butterworth(input_file: str, num_bands: int) -> tuple:
    """
    Process audio file using Butterworth bandpass filters.

    Parameters:
    input_file: str - Path to the input audio file
    num_bands: int - Number of frequency bands to create

    Returns:
    Tuple of (filtered signals, frequency ranges, sample rate, original signal)
    """
    # Read and preprocess audio
    y, sr = read_and_resample(input_file)

    # Create bandpass filters
    filters = create_bandpass_filters(num_bands, sr)

    # Apply filters to get band-limited signals
    filtered_signals = apply_filters(y, filters)

    # Calculate frequency ranges for each band
    nyquist = sr / 2
    max_freq = 0.99 * nyquist
    freq_edges = np.linspace(100, max_freq, num_bands + 1)
    ranges = {i: (freq_edges[i], freq_edges[i + 1]) for i in range(num_bands)}

    return filtered_signals, ranges, sr, y


def analyze_ci_performance(
    filtered_signals: List[np.ndarray], envelope_signals: List[np.ndarray], original_signal: np.ndarray, sr: int
) -> Dict:
    """
    Analyze the performance of the cochlear implant simulation using metrics
    relevant to speech processing and auditory perception.

    Parameters:
    filtered_signals: List of bandpass filtered signals
    envelope_signals: List of extracted envelopes
    original_signal: Original input signal
    sr: Sampling rate

    Returns:
    Dictionary containing relevant metrics
    """
    metrics = {}

    # 1. Envelope Extraction Quality
    envelope_quality = []
    for filtered, envelope in zip(filtered_signals, envelope_signals):
        # Compare with ideal envelope (using Hilbert transform)
        analytic_signal = hilbert(filtered)
        ideal_envelope = np.abs(analytic_signal)
        correlation, _ = pearsonr(ideal_envelope, envelope)
        envelope_quality.append(correlation)

    metrics["mean_envelope_correlation"] = np.mean(envelope_quality)
    metrics["min_envelope_correlation"] = np.min(envelope_quality)

    # 2. Temporal Modulation Transfer Function (TMTF)
    tmtf_metrics = calculate_tmtf(envelope_signals, sr)
    metrics.update(tmtf_metrics)

    # 3. Channel Independence
    channel_independence = calculate_channel_independence(envelope_signals)
    metrics["channel_independence"] = channel_independence

    # 4. Spectral Coverage
    spectral_metrics = analyze_spectral_coverage(filtered_signals, sr)
    metrics["spectral_coverage"] = spectral_metrics
    # metrics.update(spectral_metrics)

    return metrics


def calculate_tmtf(envelope_signals: List[np.ndarray], sr: int) -> Dict:
    """
    Calculate metrics related to the Temporal Modulation Transfer Function.
    This analyzes how well temporal modulations are preserved in different frequency bands.
    """
    tmtf_metrics = {}

    for i, envelope in enumerate(envelope_signals):
        # Calculate power spectrum of envelope
        freqs, psd = scipy.signal.welch(envelope, sr, nperseg=min(len(envelope), sr))

        # Calculate modulation index (ratio of AC to DC components)
        dc_power = psd[0]
        ac_power = np.mean(psd[1:])
        modulation_index = ac_power / dc_power if dc_power > 0 else 0

        # Store metrics for each channel
        tmtf_metrics[f"modulation_index_band_{i}"] = modulation_index

    # Calculate average modulation index across all channels
    tmtf_metrics["mean_modulation_index"] = np.mean(
        [v for k, v in tmtf_metrics.items() if k.startswith("modulation_index_band_")]
    )

    return tmtf_metrics


def calculate_channel_independence(envelope_signals: List[np.ndarray]) -> float:
    """
    Calculate how independent the different channels are from each other.
    Lower correlation between channels is generally better for CI performance.
    """
    correlations = []
    n_channels = len(envelope_signals)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            corr, _ = pearsonr(envelope_signals[i], envelope_signals[j])
            correlations.append(abs(corr))

    # Return mean absolute correlation (lower is better)
    return np.mean(correlations) if correlations else 0.0


def analyze_spectral_coverage(filtered_signals: List[np.ndarray], sr: int) -> Dict:
    """
    Analyze how well the frequency bands cover the speech-relevant spectrum.
    """
    metrics = {}

    # Define speech-relevant frequency ranges (in Hz)
    speech_ranges = {
        "f0": (50, 500),  # Fundamental frequency range
        "f1": (300, 1000),  # First formant
        "f2": (850, 2500),  # Second formant
        "consonants": (2000, 8000),  # Important for consonant discrimination
    }

    # Calculate combined coverage across all bands
    num_freq_points = 1000  # Number of points for frequency analysis
    freq_axis = np.linspace(0, sr / 2, num_freq_points)
    total_coverage = np.zeros(num_freq_points)

    for signal in filtered_signals:
        # Calculate power spectrum
        freqs, psd = scipy.signal.welch(signal, sr, nperseg=min(len(signal), sr))

        # Interpolate to common frequency axis
        interpolated_psd = np.interp(freq_axis, freqs, psd)

        # Add to total coverage (using log scale to better represent perception)
        total_coverage += np.log10(interpolated_psd + 1e-10)

    # Analyze coverage for each speech range
    for range_name, (low, high) in speech_ranges.items():
        # Calculate coverage metrics within range
        range_mask = (freq_axis >= low) & (freq_axis <= high)
        range_coverage = total_coverage[range_mask]

        if len(range_coverage) > 0:
            metrics[f"{range_name}_mean_coverage"] = np.mean(range_coverage)
            metrics[f"{range_name}_std_coverage"] = np.std(range_coverage)

            # Calculate coverage uniformity (lower std/mean ratio is more uniform)
            metrics[f"{range_name}_uniformity"] = metrics[f"{range_name}_std_coverage"] / (
                metrics[f"{range_name}_mean_coverage"] + 1e-10
            )

            # Calculate percentage of range with significant coverage
            significant_coverage = range_coverage > (np.max(range_coverage) - 20)  # -20dB threshold
            metrics[f"{range_name}_coverage_percentage"] = np.mean(significant_coverage) * 100

    # Calculate overall metrics
    metrics["total_frequency_range"] = freq_axis[-1] - freq_axis[0]

    # Calculate gaps in coverage
    threshold = np.max(total_coverage) - 20  # -20dB threshold
    gaps = total_coverage < threshold
    gap_ranges = find_gaps(freq_axis[gaps], min_gap_size=100)  # gaps larger than 100Hz
    metrics["number_of_gaps"] = len(gap_ranges)
    metrics["total_gap_width"] = sum(stop - start for start, stop in gap_ranges)

    # Calculate overlap between adjacent bands
    overlap_metric = calculate_band_overlap(filtered_signals, sr)
    metrics["band_overlap"] = overlap_metric

    return metrics


def find_gaps(freq_points: np.ndarray, min_gap_size: float) -> List[Tuple[float, float]]:
    """
    Find continuous gaps in frequency coverage above a minimum size.
    """
    if len(freq_points) == 0:
        return []

    gaps = []
    gap_start = freq_points[0]
    prev_freq = freq_points[0]

    for freq in freq_points[1:]:
        if freq - prev_freq > min_gap_size:
            gaps.append((gap_start, prev_freq))
            gap_start = freq
        prev_freq = freq

    gaps.append((gap_start, prev_freq))
    return gaps


def calculate_band_overlap(filtered_signals: List[np.ndarray], sr: int) -> float:
    """
    Calculate average overlap between adjacent frequency bands.
    """
    overlaps = []

    for i in range(len(filtered_signals) - 1):
        freqs1, psd1 = scipy.signal.welch(filtered_signals[i], sr)
        freqs2, psd2 = scipy.signal.welch(filtered_signals[i + 1], sr)

        # Interpolate to common frequency axis
        common_freqs = np.linspace(0, sr / 2, 1000)
        psd1_interp = np.interp(common_freqs, freqs1, psd1)
        psd2_interp = np.interp(common_freqs, freqs2, psd2)

        # Calculate overlap using normalized cross-correlation
        overlap = np.sum(psd1_interp * psd2_interp) / np.sqrt(np.sum(psd1_interp**2) * np.sum(psd2_interp**2))
        overlaps.append(overlap)

    return np.mean(overlaps) if overlaps else 0.0


def process_and_analyze_ci(input_file: str, num_bands: int) -> Dict:
    """
    Main function to process audio through the CI simulation and analyze performance
    """
    # Read and preprocess audio
    y, sr = read_and_resample(input_file)

    # Create and apply bandpass filters
    filtered_signals, ranges, _, _ = process_audio_with_butterworth(input_file, num_bands)

    # Extract envelopes
    rectified_signals = rectify_signals(filtered_signals)
    b_lpf, a_lpf = create_lowpass_filter(400, sr)  # 400 Hz cutoff for envelope
    envelope_signals = apply_lowpass_filter(rectified_signals, b_lpf, a_lpf)

    # Get original metrics
    metrics = analyze_ci_performance(filtered_signals, envelope_signals, y, sr)

    # Add spectral coverage metrics
    spectral_metrics = analyze_spectral_coverage(filtered_signals, sr)
    metrics.update(spectral_metrics)

    # Add basic configuration info to metrics
    metrics["num_bands"] = num_bands
    metrics["frequency_ranges"] = ranges

    return metrics


def main_ci_analysis(input_file: str, band_range: range = range(2, 31, 2)):
    """
    Run CI simulation analysis for different numbers of frequency bands
    """
    results = []

    for num_bands in band_range:
        print(f"\nProcessing with {num_bands} bands...")
        metrics = process_and_analyze_ci(input_file, num_bands)
        results.append(metrics)

    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)

    # Save results
    df.to_csv("ci_simulation_analysis.csv", index=False)

    # Create visualization of key metrics
    plot_ci_metrics(df)

    return results


def plot_ci_metrics(df: pd.DataFrame):
    """
    Create visualization of key CI simulation metrics including detailed spectral coverage analysis
    """
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    fig.suptitle("CI Simulation Performance Metrics vs Number of Frequency Bands", fontsize=16, y=0.95)

    # 1. Original metrics plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["num_bands"], df["mean_envelope_correlation"], "o-", label="Envelope Correlation")
    ax1.plot(df["num_bands"], df["channel_independence"], "s-", label="Channel Independence")
    ax1.set_xlabel("Number of Bands")
    ax1.set_ylabel("Correlation")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Signal Processing Metrics")

    # 2. Modulation Index plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["num_bands"], df["mean_modulation_index"], "o-", color="purple")
    ax2.set_xlabel("Number of Bands")
    ax2.set_ylabel("Mean Modulation Index")
    ax2.grid(True)
    ax2.set_title("Modulation Index")

    # 3. Speech Range Coverage plot
    ax3 = fig.add_subplot(gs[1, :])
    speech_ranges = ["f0", "f1", "f2", "consonants"]
    for range_name in speech_ranges:
        coverage_col = f"{range_name}_coverage_percentage"
        if coverage_col in df.columns:
            ax3.plot(df["num_bands"], df[coverage_col], "o-", label=f"{range_name}")
    ax3.set_xlabel("Number of Bands")
    ax3.set_ylabel("Coverage Percentage")
    ax3.grid(True)
    ax3.legend()
    ax3.set_title("Speech Range Coverage")

    # 4. Coverage Gaps and Overlap
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df["num_bands"], df["number_of_gaps"], "o-", label="Number of Gaps", color="red")
    ax4.set_xlabel("Number of Bands")
    ax4.set_ylabel("Number of Gaps", color="red")
    ax4.grid(True)
    ax4.tick_params(axis="y", labelcolor="red")

    ax4_twin = ax4.twinx()
    ax4_twin.plot(df["num_bands"], df["total_gap_width"], "s-", label="Total Gap Width", color="blue")
    ax4_twin.set_ylabel("Total Gap Width (Hz)", color="blue")
    ax4_twin.tick_params(axis="y", labelcolor="blue")

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax4.set_title("Coverage Gaps Analysis")

    # 5. Coverage Uniformity
    ax5 = fig.add_subplot(gs[2, 1])
    for range_name in speech_ranges:
        uniformity_col = f"{range_name}_uniformity"
        if uniformity_col in df.columns:
            ax5.plot(df["num_bands"], df[uniformity_col], "o-", label=f"{range_name}")
    ax5.set_xlabel("Number of Bands")
    ax5.set_ylabel("Coverage Uniformity")
    ax5.grid(True)
    ax5.legend()
    ax5.set_title("Coverage Uniformity by Speech Range")

    plt.tight_layout()
    plt.savefig("ci_simulation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


# def plot_ci_metrics(df: pd.DataFrame):
#     """
#     Create visualization of key CI simulation metrics
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle("CI Simulation Performance Metrics vs Number of Frequency Bands")

#     # Plot key metrics
#     metrics_to_plot = [
#         ("mean_envelope_correlation", "Mean Envelope Correlation"),
#         ("channel_independence", "Channel Independence"),
#         ("mean_modulation_index", "Mean Modulation Index"),
#         ("tmtf_data", "TMTF Data"),
#         ("spectral_coverage", "Spectral Coverage"),
#     ]

#     for i, (metric, title) in enumerate(metrics_to_plot):
#         ax = axes.flat[i]
#         ax.plot(df["num_bands"], df[metric], "o-")
#         ax.set_xlabel("Number of Bands")
#         ax.set_ylabel(title)
#         ax.grid(True)

#     plt.tight_layout()
#     plt.savefig("ci_simulation_metrics.png")
#     plt.close()


if __name__ == "__main__":
    # Configuration
    files = ["../data/fox_white_noise.wav"]  # Add your audio file paths here
    num_bands = 8
    cutoff_freq = 400  # Hz

    for fp in files:
        results = main_ci_analysis(fp)
        # for result in all_results:
        #     num_bands = result["num_bands"]
        #     # Process with the best number of bands
        #     filtered_signals, ranges, sr, original = process_audio_with_butterworth(fp, num_bands)

        #     # Save individual band signals
        #     save_band_signals(filtered_signals, sr, fp, f"butterworth_output/{num_bands}_output")

        #     # Save combined signal
        #     combined = combine_bands(filtered_signals)
        #     save_band_signals([combined], sr, fp, f"butterworth_output/{num_bands}_output", "butterworth_combined.wav")

        #     # Plot some analysis for the best configuration
        #     # Plot the lowest and highest frequency channel outputs
        #     plot_signal(filtered_signals[0], sr, f"butterworth_output/{num_bands}_output/lowest_freq_band")
        #     plot_signal(filtered_signals[-1], sr, f"butterworth_output/{num_bands}_output/highest_freq_band")

        #     # Create envelope signals
        #     rectified_signals = rectify_signals(filtered_signals)
        #     b_lpf, a_lpf = create_lowpass_filter(400, sr)  # 400 Hz cutoff
        #     envelope_signals = apply_lowpass_filter(rectified_signals, b_lpf, a_lpf)

        #     # Plot envelopes
        #     plot_signal(envelope_signals[0], sr, f"butterworth_output/{num_bands}_output/lowest_freq_envelope")
        #     plot_signal(envelope_signals[-1], sr, f"butterworth_output/{num_bands}_output/highest_freq_envelope")
        # # Phase 1: Read and resample audio
    # y, sr = read_and_resample(fp)

    # # Phase 2: Bandpass filter bank creation
    # filters = create_bandpass_filters(num_bands, sr)

    # # Apply bandpass filters to the audio signal
    # filtered_signals = apply_filters(y, filters)

    # combined_band = combine_bands(filtered_signals)

    # save_band_signals(filtered_signals, sr, fp, "bandpass_output")

    # # Combine bands and save (to verify they are split correctly)

    # save_band_signals([combined_band], sr, fp, "output", "bandpass_combined.wav")

    # # Plot the lowest and highest frequency channel outputs
    # plot_signal(filtered_signals[0], sr, f"output/Lowest frequency channel output_{fp}")
    # plot_signal(filtered_signals[-1], sr, f"output/Highest frequency channel output_{fp}")

    # # Rectify the filtered signals
    # rectified_signals = rectify_signals(filtered_signals)

    # # Create and apply a lowpass filter for envelope extraction
    # b_lpf, a_lpf = create_lowpass_filter(cutoff_freq, sr)
    # envelope_signals = apply_lowpass_filter(rectified_signals, b_lpf, a_lpf)

    # # Plot the extracted envelope of the lowest and highest frequency channels
    # plot_signal(envelope_signals[0], sr, f"output/Extracted envelope of the lowest frequency channel_{fp}")
    # plot_signal(envelope_signals[-1], sr, f"output/Extracted envelope of the highest frequency channel_{fp}")

    # # Save the processed audio outputs (optional)
    # save_audio(envelope_signals[0], sr, f"output/envelope_lowest_channel_{fp}")
    # save_audio(envelope_signals[-1], sr, f"output/envelope_highest_channel_{fp}")
