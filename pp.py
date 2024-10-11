import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def read_and_resample(file_path):
    # Read the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Check if stereo or mono
    channels = "Stereo" if y.ndim > 1 and y.shape[0] == 2 else "Mono"

    print(f"File: {file_path}")
    print(f"Channels: {channels}")
    print(f"Original sampling rate: {sr} Hz")

    # Resample to 16 kHz if necessary
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        print("Resampled to 16 kHz")

    return y, sr


# plot_waveform generates and saves the waveform of the file as a function of its sample number
def plot_waveform(y, sr, title):
    sample_numbers = np.arange(len(y))
    plt.figure(figsize=(12, 6))
    plt.plot(sample_numbers, y)
    plt.title(f"{title} waveform as a function of sample number")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()


def calculate_frequency_representation_score(original_signal, processed_signal, sr, low_freq_range, high_freq_range):
    S_orig = np.abs(librosa.stft(original_signal)) ** 2
    S_proc = np.abs(librosa.stft(processed_signal)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(S_orig)
    print("total energy", total_energy)
    low_band_energy = np.sum(S_proc[(freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1])])
    high_band_energy = np.sum(S_proc[(freqs >= high_freq_range[0]) & (freqs <= high_freq_range[1])])
    low_accuracy = (low_band_energy / total_energy) * 100
    high_accuracy = (high_band_energy / total_energy) * 100
    frequency_representation_score = (low_accuracy + high_accuracy) / 2
    return high_accuracy, low_accuracy, frequency_representation_score


def energy(signal, range, sr):
    # Compute the Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(signal)) ** 2
    # Get the frequency bins
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= range[0]) & (freqs <= range[1])
    return np.sum(S[mask])


def simulate_cochlear_implant(original_signal, sr, num_channels=22):
    # Define frequency ranges for each channel
    # Cochlear implants typically cover a range from about 100 Hz to 8000 Hz
    min_freq, max_freq = 100, 8000
    channel_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_channels + 1)

    # Compute spectrogram
    S = librosa.stft(original_signal)

    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr)

    # Initialize output spectrogram
    S_processed = np.zeros_like(S)

    # Process each channel
    for i in range(num_channels):
        lower, upper = channel_edges[i], channel_edges[i + 1]

        # Find frequency bins within this channel
        channel_mask = (freqs >= lower) & (freqs < upper)

        if np.any(channel_mask):
            # Calculate the average magnitude for this channel
            channel_mag = np.mean(np.abs(S[channel_mask, :]), axis=0)

            # Apply this magnitude to all frequencies in the channel
            S_processed[channel_mask, :] = channel_mag * np.exp(1j * np.angle(S[channel_mask, :]))

    # Inverse STFT to get the processed signal
    y_processed = librosa.istft(S_processed)

    # Ensure the processed signal has the same length as the original
    y_processed = librosa.util.fix_length(y_processed, size=len(original_signal))

    return y_processed


# Example usage
original_signal, sr = read_and_resample("with_rain.wav")
mock_processed_signal = simulate_cochlear_implant(original_signal, sr)

low_freq_range = (0, 500)  # Low frequency range
high_freq_range = (2000, 4000)  # High frequency range

# original signal energies
original_low_energy = energy(original_signal, low_freq_range, sr)
original_high_energy = energy(original_signal, high_freq_range, sr)

# processed signal energies
processed_low_energy = energy(mock_processed_signal, low_freq_range, sr)
processed_high_energy = energy(mock_processed_signal, high_freq_range, sr)

print("Original (low,high)")
print(f"low: {original_low_energy} | high: {original_high_energy}")

print("processed (low,high)")
print(f"low: {processed_low_energy} | high: {processed_high_energy}")


low_accuracy = 100 - (
    abs(original_low_energy - processed_low_energy) / ((original_low_energy + processed_low_energy) / 2) * 100
)
high_accuracy = 100 - (
    abs(original_high_energy - processed_high_energy) / ((original_high_energy + processed_high_energy) / 2) * 100
)

print("low accuracy: ", low_accuracy)
print("high accuracy: ", high_accuracy)
