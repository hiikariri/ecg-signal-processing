import numpy as np
import matplotlib.pyplot as plt

fs = 1000
N = 2000

first_amplitude = int(input("Enter A1: "))
first_freq = int(input("Enter F1: "))
second_amplitude = int(input("Enter A2: "))
second_freq = int(input("Enter F2: "))
mean = int(input("Enter mean: "))
deviation = int(input("Enter deviation: "))
N = int(input("Enter the number of data points(def = 2000): "))
fs = int(input("Enter the sampling frequency (def = 1000): "))


x = np.zeros(N)  # TODO input signal processing
x_real = np.zeros(N)
x_im = np.zeros(N)
MagDFT = np.zeros(N)

for n in range(N):
    x[n] = first_amplitude*np.sin(2*np.pi*first_freq*n/fs) + \
        second_amplitude*np.sin(2*np.pi*second_freq*n/fs)

for k in range(N):
    for n in range(N):
        x_real[k] += x[n]*np.cos(2*np.pi*k*n/N)
        x_im[k] -= x[n]*np.sin(2*np.pi*k*n/N)

for k in range(N):
    MagDFT[k] = np.sqrt(np.power(x_real[k], 2) + np.power(x_im[k], 2))

n = np.arange(0, N, dtype=int)
k = np.arange(0, N, dtype=int)

# plotting sinyal input
print('Input Signal :')
plt.figure(figsize=((10, 5)))
plt.stem(n/fs, x[n])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Domain")
plt.show()

# plotting sinyal DFT
print('DFT :')
plt.figure(figsize=((10, 5)))
plt.stem(k*fs/N, MagDFT[k])
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain")
plt.show()
