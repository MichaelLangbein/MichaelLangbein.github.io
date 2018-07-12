import numpy as np
import matplotlib.pyplot as plt


def signal(t):
    s = np.sin( 440.0 * 2.0 * np.pi * t)
    return s


def addOctave(ampls):
    return ampls


deltaT = 0.001
samplePoints = [i*deltaT for i in range(500)]
samples = signal(samplePoints)

plt.plot(samplePoints, samples)
plt.show()

ampls = np.fft.fft(samples)
amplsNew = addOctave(ampls)
samplesNew = np.fft.ifft(amplsNew)

plt.plot(samplePoints, samplesNew)
plt.show()
