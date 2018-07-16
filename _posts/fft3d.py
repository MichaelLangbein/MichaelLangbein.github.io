import numpy as np
import matplotlib.pyplot as plt

def fft(data):
    dataX = data[:,0]
    dataY = data[:,1]
    dataZ = data[:,2]
    ampsX = np.fft.fft(dataX)
    ampsY = np.fft.fft(dataY)
    ampsZ = np.fft.fft(dataZ)
    amps = np.zeros(np.shape(data), dtype=np.complex128)
    amps[:,0] = ampsX
    amps[:,1] = ampsY
    amps[:,2] = ampsZ
    return amps

def ifft(amps):
    ampsX = amps[:,0]
    ampsY = amps[:,1]
    ampsZ = amps[:,2]
    dataX = np.fft.ifft(ampsX)
    dataY = np.fft.ifft(ampsY)
    dataZ = np.fft.ifft(ampsZ)
    data = np.zeros(np.shape(amps))
    data[:,0] = dataX
    data[:,1] = dataY
    data[:,2] = dataZ
    return data

r1 = r2 = r3 = 1
def signal(theta, phi):
    x = r1 * np.sin(phi) * np.cos(theta)
    y = r2 * np.sin(phi) * np.sin(theta)
    z = r3 * np.cos(phi)
    return [x, y, z]

def getSamplePoints(deltaTheta, deltaPhi, nTheta, nPhi):
    samplePoints = np.zeros([nTheta * nPhi, 2])
    r = 0
    for t in range(nTheta):
        for p in range(nPhi):
            samplePoints[r,:] = [t*deltaTheta, p*deltaPhi]
            r += 1
    return samplePoints

def getSamples(samplePoints):
    rows, cols = np.shape(samplePoints)
    samples = np.zeros([rows, 3])
    for rowNr, rowData in enumerate(samplePoints):
        samples[rowNr,:] = signal(rowData[0], rowData[1])
    return samples


def addOctave(ampls):
    return ampls


def plotSamples(samplePoints, samples):
    pass
    

deltaTheta = 1
deltaPhi = 1
samplePoints = getSamplePoints(deltaTheta, deltaPhi, 400, 400)
samples = getSamples(samplePoints)

plotSamples(samplePoints, samples)

ampls = fft(samples)
amplsNew = addOctave(ampls)
samplesNew = ifft(amplsNew)

plotSamples(samplePoints, samplesNew)
