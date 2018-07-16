import numpy as np
import matplotlib.pyplot as plt


def getMax(data):
    posMax = 0
    valMax = data[0]
    for pos, val in enumerate(data):
        if abs(val) > abs(valMax):
            posMax = pos
            valMax = val
    return (posMax, valMax)


def signal(t):
    s = np.sin( 440.0 * 2.0 * np.pi * t)
    return s


def addOctave(ampls, deltaT):
    indxMax, valMax = getMax(ampls)
    indxOctHigher = 2*indxMax
    while indxOctHigher > len(ampls):
        indxOctHigher -= len(ampls)
    print( "Index max ampl: {}, index oct on top: {}, len data: {}".format(indxMax, indxOctHigher, len(ampls)))
    ampls[indxOctHigher] = ampls[indxMax]
    return ampls


deltaT = 0.00001
samplePoints = [i*deltaT for i in range(1000)]
samples = [signal(t) for t in samplePoints]

plt.plot(samplePoints, samples)
plt.show()

ampls = np.fft.fft(samples)
amplsNew = addOctave(ampls, deltaT)
samplesNew = np.fft.ifft(amplsNew)

plt.plot(samplePoints, samplesNew)
plt.show()
