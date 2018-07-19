import numpy as np
import matplotlib . pyplot as plt
from mpl_toolkits . mplot3d import Axes3D


r1 = r2 = r3 = 1
def body(theta, phi):
    x = r1 * np.cos(theta) * np.sin(phi)
    y = r2 * np.sin(theta) * np.sin(phi)
    z = r3 * np.cos(phi)
    return [x, y, z]

def getSample(thetas, phis):
    samples = np.zeros((len(thetas), len(phis), 3))
    for r, theta in enumerate(thetas):
        for c, phi in enumerate(phis):
            samples[r, c, :] = body(theta, phi)
    return samples

def fft(samples):
    amps = np.zeros(np.shape(samples), dtype=np.complex128)
    amps[:,:,0] = np.fft.fft2(samples[:,:,0])
    amps[:,:,1] = np.fft.fft2(samples[:,:,1])
    amps[:,:,2] = np.fft.fft2(samples[:,:,2])
    return amps

def addOctaveSingle(amps):
    lenTheta, lenPhi = amps.shape
    indxMaxTheta, indxMaxPhi = np.unravel_index(amps.argmax(), amps.shape)
    indxMaxThetaOct = (2*indxMaxTheta)%lenTheta
    indxMaxTPhiOct = (2*indxMaxPhi)%lenPhi
    amps[indxMaxThetaOct, indxMaxTPhiOct] *= 2
    return amps

def addOctave(amps):
    amps[:,:,0] = addOctaveSingle(amps[:,:,0])
    amps[:,:,1] = addOctaveSingle(amps[:,:,1])
    amps[:,:,2] = addOctaveSingle(amps[:,:,2])
    return amps

def ifft(amps):
    samples = np.zeros(np.shape(amps))
    samples[:,:,0] = np.fft.ifft2(amps[:,:,0])
    samples[:,:,1] = np.fft.ifft2(amps[:,:,1])
    samples[:,:,2] = np.fft.ifft2(amps[:,:,2])
    return samples

def plot(samples):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for row in samples:
        for item in row:
            ax.scatter(item[0],item[1],item[2])
    plt.show()


thetas = np.linspace(0, 720, 40)
phis = np.linspace(0, 720, 40)
sample = getSample(thetas, phis)
plot(sample)
amps = fft(sample)
ampsNew = addOctave(amps)
sampleNew = ifft(ampsNew)
plot(sampleNew)
