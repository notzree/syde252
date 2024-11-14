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


def analyze_butterworth_performance(input_file: str, band_range: range) -> Tuple[Dict, List[Dict]]:
    """
    Analyze audio processing performance with different numbers of Butterworth bandpass filters.

    Parameters:
    input_file: str - Path to the input audio file
    band_range: range - Range of number of bands to test

    Returns:
    Tuple containing best results and all results
    """
    results = []
    best_result = {"num_bands": 0, "metrics": None, "score": float("inf")}

    for num_bands in band_range:
        print(f"\nProcessing with {num_bands} Butterworth bands...")

        # Process audio with current number of bands
        filtered_signals, ranges, sr, original_signal = process_audio_with_butterworth(input_file, num_bands)

        # Combine bands
        combined_signal = combine_bands(filtered_signals)

        # Compare signals
        comparison = compare_signals(original_signal, combined_signal, sr)

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


def main_butterworth_analysis(fp):
    """
    Main function to run the Butterworth filter analysis
    """
    input_file = fp
    band_range = range(2, 21, 2)  # Test even numbers from 2 to 20 bands

    # Run analysis
    best_result, all_results = analyze_butterworth_performance(input_file, band_range)

    # Print best result
    print("\nBest Performance with Butterworth Filters:")
    print(f"Number of bands: {best_result['num_bands']}")
    print("\nMetrics for best configuration:")
    print_comparison_results(best_result["metrics"])

    # Create and save performance plot
    fig = plot_performance_metrics(all_results)
    plt.savefig("butterworth_band_performance.png")
    plt.close()

    # Save results to CSV
    # Prepare data for DataFrame
    df_data = []
    for result in all_results:
        row = {"num_bands": result["num_bands"]}
        row.update(result["metrics"])
        df_data.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(df_data)
    df.to_csv("butterworth_analysis_results.csv", index=False)

    return best_result, all_results
