import numpy as np
import matplotlib.pyplot as plt
import os

def plot(x_axis, data, title, xlabel, ylabel, label = 'Series', subplot_position = 221, color='blue', stem=False):
    plt.subplot(subplot_position)
    if stem:
        plt.stem(x_axis, data, label=label,
                 linefmt=color, use_line_collection=True)
        plt.xlim(0, sampling_freq/2)
    else:
        plt.plot(x_axis, data, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


def dft_by_part(start_sample, end_sample, data, replicate_coeff, sampling_freq, name):
    n_start = int(sampling_freq * start_sample)
    n_end = int(sampling_freq * end_sample)
    new_data = data[n_start:n_end]
    # Replicate the wave to make a periodic signal
    periodic_signal = np.tile(new_data, replicate_coeff)
    total_data_periodic = len(periodic_signal)

    plt.figure()
    plot(np.arange(0, total_data_periodic), periodic_signal, 'Sample ' + name, 'Sequence', 'Amplitude', color='red')

    # Frequency axis for the periodic signal
    freq_periodic = np.arange(0, total_data_periodic) * sampling_freq / total_data_periodic

    # Calculate the DFT for the periodic signal
    MagDFT_periodic = dft(total_data_periodic, periodic_signal)
    return MagDFT_periodic, freq_periodic


def dft(total_data, data):
    x_real = np.zeros(total_data)  # Initialize the real part of the DFT
    x_im = np.zeros(total_data)  # Initialize the imaginary part of the DFT
    MagDFT = np.zeros(total_data)  # Initialize the magnitude of the DFT

    for k in range(total_data):
        for n in range(total_data):
            # Calculate the real part of the DFT
            x_real[k] += data[n] * np.cos(2 * np.pi * k * n / total_data)
            # Calculate the imaginary part of the DFT
            x_im[k] -= data[n] * np.sin(2 * np.pi * k * n / total_data)

    for k in range(total_data):
        # Calculate the magnitude of the DFT
        MagDFT[k] = np.sqrt(x_real[k]**2 + x_im[k]**2)

    return MagDFT


if __name__ == "__main__":
    # Load the data from the files
    try:
        print("Loading data...")
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)
        file_davis = os.path.join(script_dir, 'Data ECG Davis.txt')
        file_levy = os.path.join(script_dir, 'Data ECG Levy.txt')
        sequence_davis, data_davis = np.loadtxt(
            file_davis, skiprows=1, delimiter=None, unpack=True)
        sequence_levy, data_levy = np.loadtxt(
            file_levy, skiprows=1, delimiter=None, unpack=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    sampling_freq = 250  # sampling frequency in Hz

    total_data_davis = len(sequence_davis)
    total_data_levy = len(sequence_levy)

    # Total sampling time
    sampling_time_davis = total_data_davis / sampling_freq
    sampling_time_levy = total_data_levy / sampling_freq

    # Time axis
    time_davis = np.arange(0, sampling_time_davis, 1/sampling_freq)
    time_levy = np.arange(0, sampling_time_levy, 1/sampling_freq)

    # Frequency axis
    freq_davis = np.arange(0, total_data_davis) * sampling_freq / total_data_davis
    freq_levy = np.arange(0, total_data_levy) * sampling_freq / total_data_levy

    # Calculate the DFT for both datasets
    MagDFT_davis = dft(total_data_davis, data_davis)
    MagDFT_levy = dft(total_data_levy, data_levy)

    # Isolate and replicate parts of the ECG signals
    davis_p_wave, davis_p_wave_freq = dft_by_part(0.1, 0.2, data_davis, 5, sampling_freq, 'P Davis')
    davis_qrs_wave, davis_qrs_wave_freq = dft_by_part(0.2, 0.3, data_davis, 5, sampling_freq, 'QRS Davis')
    davis_t_wave, davis_t_wave_freq = dft_by_part(0.3, 0.4, data_davis, 5, sampling_freq, 'T Davis')

    levy_p_wave, levy_p_wave_freq = dft_by_part(0.65, 0.73, data_levy, 5, sampling_freq, 'P Levy')
    levy_qrs_wave, levy_qrs_wave_freq = dft_by_part(0.73, 0.8, data_levy, 5, sampling_freq, 'QRS Levy')
    levy_t_wave, levy_t_wave_freq = dft_by_part(0.8, 0.9, data_levy, 5, sampling_freq, 'T Levy')

    # Plot initialization
    plt.figure(figsize=(16, 10))

    # Plot ECG signal
    plot(time_davis, data_davis, 'Davis ECG Signal', 'Time (s)', 'Voltage', 'ECG', 221)
    plot(time_levy, data_levy, 'Levy ECG Signal', 'Time (s)', 'Voltage', 'ECG', 222, color='orange')

    # Plot DFT
    plot(freq_davis, MagDFT_davis, 'DFT of Davis ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 223, stem=True)
    plot(freq_levy, MagDFT_levy, 'DFT of Levy ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 224, color='orange', stem=True)

    plt.figure(figsize=(16, 10))

    # Plot DFT P wave
    plot(davis_p_wave_freq, davis_p_wave, 'DFT of Davis P ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 221, stem=True)
    plot(levy_p_wave_freq, levy_p_wave, 'DFT of Levy P ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 222, color='orange', stem=True)

    # Plot DFT QRS wave
    plot(davis_qrs_wave_freq, davis_qrs_wave, 'DFT of Davis QRS ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 223, stem=True)
    plot(levy_qrs_wave_freq, levy_qrs_wave, 'DFT of Levy QRS ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 224, color='orange', stem=True)

    plt.figure(figsize=(16, 10))
    # Plot DFT T wave
    plot(davis_t_wave_freq, davis_t_wave, 'DFT of Davis T ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 221, stem=True)
    plot(levy_t_wave_freq, levy_t_wave, 'DFT of Levy T ECG Signal', 'Frequency (Hz)', 'Magnitude', 'DFT', 222, color='orange', stem=True)

    # Show plot
    plt.tight_layout()
    plt.show()
