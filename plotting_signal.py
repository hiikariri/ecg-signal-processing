import matplotlib.pyplot as plt
import numpy as np

A = (float)(input("Amplitude: "))
n = (int)(input("Number of samples: "))
fs = (int)(input("Sampling frequency: "))

time = np.zeros(n)
y = np.zeros(n)

for i in range(n):
    time[i] = i/fs
    y[i] = 2 * np.sin(130 * (np.pi) * time[i])

plt.plot(time, y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sine wave')
plt.show()
