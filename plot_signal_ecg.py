import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('Data ECG Davis.txt')

plt.plot(data[1:])
plt.show()
