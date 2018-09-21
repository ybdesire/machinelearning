from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import numpy as np

[Fs, x] = audioBasicIO.readAudioFile("happy.wav");
# Fs is frequency
# x is real data

time = np.arange(0,len(x))*1.0/Fs
plt.plot(time, x)
plt.show()