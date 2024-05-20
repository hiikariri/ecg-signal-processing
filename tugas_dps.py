import numpy as np  # Import numpy library
import matplotlib.pyplot as plt  # Import matplotlib.pyplot library

amplitude = int(input('Enter the amplitude: '))  # Input the amplitude
sampling_freq = int(input('Enter the sampling frequency: ')
                    )  # Input the sampling frequency
sampling_time = int(input('Enter the sampling time: ')
                    )  # Input the sampling time

total_data = sampling_time * sampling_freq  # Calculate the total data
time = np.arange(0, sampling_time, 1/sampling_freq)  # Create the time axis
y = np.zeros(len(time))  # initialize the signal output
freq = np.piecewise(time,
                    [(time < 5), ((time >= 5) & (time < 10)), (time >= 10)],
                    [2, 10, 50])  # just a one-liner if-esle statement to create the frequency axis

y = amplitude * np.cos(2 * np.pi * freq * time)  # Create the signal

# Plot the time domain signal
plt.figure(figsize=((10, 5)))

plt.subplot(2, 1, 1)
plt.plot(time, y, label='Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal')
plt.legend()

x_real = np.zeros(total_data)  # Initialize the real part of the DFT
x_im = np.zeros(total_data)  # Initialize the imaginary part of the DFT
MagDFT = np.zeros(total_data)  # Initialize the magnitude of the DFT

for k in range(total_data):
    for n in range(total_data):
        # Calculate the real part of the DFT
        x_real[k] += y[n]*np.cos(2*np.pi*k*n/total_data)
        # Calculate the imaginary part of the DFT
        x_im[k] -= y[n]*np.sin(2*np.pi*k*n/total_data)

for k in range(total_data):
    # Calculate the magnitude of the DFT
    MagDFT[k] = np.sqrt(x_real[k]**2 + x_im[k]**2)

freq_axis = np.arange(0, total_data) * sampling_freq / total_data  # Create the frequency axis

# Plot the DFT result
plt.subplot(2, 1, 2)
plt.stem(freq_axis, MagDFT, use_line_collection=True, label='DFT')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("DFT Result")
plt.xlim(0, sampling_freq/2)  # Only show positive frequencies
plt.legend()

plt.tight_layout()
plt.show()
