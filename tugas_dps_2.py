import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.interpolate import make_interp_spline


def moving_average(data, orde_filter):
    y = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(orde_filter):
            if i - j >= 0:
                y[i] += data[i - j]
        y[i] /= orde_filter
    return y


def low_pass_filter(cutoff_freq, index, sampling_freq, data):
    ohm = 2 * np.pi * cutoff_freq / sampling_freq
    h = np.zeros(len(index))
    y = np.zeros(len(data))
    for i in range(len(index)):
        if index[i] == 0:
            h[i] = ohm / np.pi
        else:
            h[i] = np.sin(ohm * index[i]) / (np.pi * index[i])
    for n in range(len(data)):
        for i in range(len(index)):
            if n - i >= 0:
                y[n] += h[i] * data[n - i]

    return y


def band_pass_filter(low_cutoff_freq, high_cutoff_freq, index, sampling_freq, data):
    y = np.zeros(len(data))
    low_cutoff_ohm = 2 * np.pi * low_cutoff_freq / sampling_freq
    high_cutoff_ohm = 2 * np.pi * high_cutoff_freq / sampling_freq
    h = np.zeros(len(index))
    for i in range(len(index)):
        if index[i] == 0:
            h[i] = (high_cutoff_ohm - low_cutoff_ohm) / np.pi
        else:
            h[i] = (np.sin(high_cutoff_ohm * index[i]) -
                    np.sin(low_cutoff_ohm * index[i])) / (np.pi * index[i])
    for n in range(len(data)):
        for i in range(len(index)):
            if n - i >= 0:
                y[n] += h[i] * data[n - i]
    return y


def filter_by_scratch(data, sampling_freq, cutoff_freq=0, filter_order=1, low_cutoff_freq=0, high_cutoff_freq=0, filter_type="low_pass"):
    M = (filter_order - 1) // 2
    index = np.arange(-M, M + 1, 1, dtype=int)
    match filter_type:
        case "low_pass":
            print("YES LOW PASS")
            filtered_signal = low_pass_filter(
                cutoff_freq, index, sampling_freq, data)
        case "high_pass":
            # filtered_signal = high_pass_filter(cutoff_freq, M, index, sampling_freq, data)
            return
        case "band_pass":
            print("YES BAND PASS")
            filtered_signal = band_pass_filter(
                low_cutoff_freq, high_cutoff_freq, index, sampling_freq, data)
        case "band_stop":
            # filtered_signal = band_stop_filter(cutoff_freq, M, index, sampling_freq, data)
            return
        case _:
            raise ValueError("Invalid filter type")

    return filtered_signal


def low_pass_filter_by_library(data, cutoff_freq, filter_order):
    b, a = signal.butter(filter_order, cutoff_freq, 'low', analog=False)
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal


def spline_interpolate(x_source, y_source):
    spline = make_interp_spline(x_source, y_source, k=3)
    x_smooth = np.linspace(x_source.min(), x_source.max(), 300)
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth


def plot(x_axis, data, title, xlabel, ylabel, label='Series', subplot_position=221, color='blue', stem=False, half_freq=False):
    plt.subplot(subplot_position)
    if stem:
        plt.stem(x_axis, data, label=label,
                 linefmt=color, use_line_collection=True)
        plt.xlim(0, sampling_freq/2)
    else:
        plt.plot(x_axis, data, label=label, color=color)
        if half_freq:
            plt.xlim(0, sampling_freq/2)
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
    plot(np.arange(0, total_data_periodic), periodic_signal,
         'Sample ' + name, 'Sequence', 'Amplitude', color='red')

    # Frequency axis for the periodic signal
    freq_periodic = np.arange(0, total_data_periodic) * \
        sampling_freq / total_data_periodic

    # Calculate the DFT for the periodic signal
    dft_magnitude_periodic = dft(total_data_periodic, periodic_signal)
    return dft_magnitude_periodic, freq_periodic


def dft(total_data, data):
    x_real = np.zeros(total_data)
    x_im = np.zeros(total_data)
    dft_magnitude = np.zeros(total_data)

    for k in range(total_data):
        for n in range(total_data):
            x_real[k] += data[n] * np.cos(2 * np.pi * k * n / total_data)
            x_im[k] -= data[n] * np.sin(2 * np.pi * k * n / total_data)

    for k in range(total_data):
        dft_magnitude[k] = np.sqrt(x_real[k]**2 + x_im[k]**2)

    return dft_magnitude


if __name__ == "__main__":
    orde = int(input("Enter the filter order: "))
    path = input("Enter the path of the data: ")
    sampling_freq = int(input("Enter the sampling frequency: "))
    try:
        print("Loading data...")
        script_dir = os.path.dirname(__file__)
        data_path = os.path.join(script_dir, 'Data ECG Davis.txt')
        sequence, data = np.loadtxt(
            data_path, skiprows=1, delimiter=None, unpack=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    total_data = len(sequence)
    # Total sampling time
    sampling_time = total_data / sampling_freq
    # Time axis
    time_axis = np.arange(0, sampling_time, 1/sampling_freq)

    plt.figure(figsize=(16, 10))
    data = data - np.mean(data)  # TODO: USE POLYNOMIAL REGRESSION
    plot(time_axis, data, 'Raw ECG Signal', 'Time (s)',
         'Voltage', 'ECG', 221, color='red')

    data_filtered = filter_by_scratch(
        data=data, cutoff_freq=40, filter_order=orde, sampling_freq=125)
    plot(time_axis, data_filtered, 'Filtered ECG Signal',
         'Time (s)', 'Voltage', 'ECG filtered', 222, color='blue')

    data_filtered = filter_by_scratch(
        data=data_filtered[::-1], cutoff_freq=40, filter_order=orde, sampling_freq=125)[::-1]
    plot(time_axis, data_filtered, 'Filtered ECG Signal Backward',
         'Time (s)', 'Voltage', 'ECG filtered backward', 223, color='green')

    data_filtered = moving_average(data_filtered, orde)
    plot(time_axis, data_filtered, 'MAV ECG Signal', 'Time (s)',
         'Voltage', 'ECG after MAV', 224, color='pink')

    plt.figure(figsize=(16, 10))
    segmented_data = filter_by_scratch(data=data_filtered, filter_order=orde, sampling_freq=125,
                                       filter_type="band_pass", low_cutoff_freq=8, high_cutoff_freq=23)
    plot(time_axis, segmented_data, 'QRS Segmentation ECG',
         'Time (s)', 'Voltage', 'QRS', 221, color='blue')

    # Frequency axis
    freq_axis = np.arange(0, total_data) * sampling_freq / total_data

    # Calculate the DFT for both datasets
    dft_magnitude = dft(total_data, data)
    dft_magnitude_filtered = dft(total_data, data_filtered)
    dft_magnitude_segmented = dft(total_data, segmented_data)

    # Isolate and replicate parts of the ECG signals
    p_wave, p_wave_freq = dft_by_part(
        1.185, 1.263, data_filtered, 1, sampling_freq, 'P')
    qrs_wave, qrs_wave_freq = dft_by_part(
        1.279, 1.420, data_filtered, 1, sampling_freq, 'QRS')
    t_wave, t_wave_freq = dft_by_part(
        1.555, 1.616, data_filtered, 1, sampling_freq, 'T')

    p_wave_freq_smooth, p_wave_smooth = spline_interpolate(p_wave_freq, p_wave)
    qrs_wave_freq_smooth, qrs_wave_smooth = spline_interpolate(
        qrs_wave_freq, qrs_wave)
    t_wave_freq_smooth, t_wave_smooth = spline_interpolate(t_wave_freq, t_wave)

    # Plot initialization
    plt.figure(figsize=(16, 10))

    # Plot DFT
    plot(freq_axis, dft_magnitude, 'DFT of the ECG Signal',
         'Frequency (Hz)', 'Magnitude', 'DFT', 221, stem=True)
    plot(freq_axis, dft_magnitude_filtered, 'DFT of Filtered ECG Signal',
         'Frequency (Hz)', 'Magnitude', 'DFT', 222, stem=True)
    plot(freq_axis, dft_magnitude_segmented, 'DFT of segmented ECG Signal',
         'Frequency (Hz)', 'Magnitude', 'DFT', 223, stem=True)

    plt.figure(figsize=(16, 10))

    plot(p_wave_freq, p_wave, 'DFT of P ECG Signal', 'Frequency (Hz)',
         'Magnitude', 'P', 221, stem=False, half_freq=True)
    plot(qrs_wave_freq, qrs_wave, 'DFT of QRS ECG Signal', 'Frequency (Hz)',
         'Magnitude', 'QRS', 221, color='orange', stem=False, half_freq=True)
    plot(t_wave_freq, t_wave, 'DFT of T ECG Signal', 'Frequency (Hz)',
         'Magnitude', 'DFT by Part', 221, color='green', stem=False, half_freq=True)

    plot(p_wave_freq_smooth, p_wave_smooth, 'DFT of P ECG Signal',
         'Frequency (Hz)', 'Magnitude', 'P', 222, stem=False, half_freq=True)
    plot(qrs_wave_freq_smooth, qrs_wave_smooth, 'DFT of QRS ECG Signal', 'Frequency (Hz)',
         'Magnitude', 'QRS', 222, color='orange', stem=False, half_freq=True)
    plot(t_wave_freq_smooth, t_wave_smooth, 'DFT of T ECG Signal', 'Frequency (Hz)',
         'Magnitude', 'DFT by Part', 222, color='green', stem=False, half_freq=True)

    # Show plot
    plt.tight_layout()
    plt.show()
