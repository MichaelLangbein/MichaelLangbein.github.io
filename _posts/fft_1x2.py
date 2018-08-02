import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

r1 = 3
r2 = 1
def body(t):
    x = r1 * np.cos(t)
    y = r2 * np.sin(t)
    return [x, y]


def getSample(ts):
    samples = np.zeros((len(ts), 2))
    for r, t in enumerate(ts):
        samples[r, :] = body(t)
    return samples


def fft(samples):
    amps = np.zeros(np.shape(samples), dtype=np.complex128)
    amps[:,0] = np.fft.fft(samples[:,0])
    amps[:,1] = np.fft.fft(samples[:,1])
    return amps

def ifft(amps):
    samples = np.zeros(np.shape(amps), dtype=amps.dtype)
    samples[:,0] = np.fft.ifft(amps[:,0])
    samples[:,1] = np.fft.ifft(amps[:,1])
    return np.real(samples)


def addOctave(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,0] = addOctaveSingle(amps[:,0])
    ampsNew[:,1] = addOctaveSingle(amps[:,1])
    return ampsNew

def addOctaveSingle(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    R = len(amps)
    for r, val in enumerate(amps):
        ampsNew[r] += val
        if r < R//4:
            r2 = 2*r
            ampsNew[r2] += val
        elif r > (3.0 * R // 4):
            dist = R-r
            r2 = R-(2*dist)
            ampsNew[r2] += val
    return ampsNew

def filterAmps(amps, perc):
    ampsF = np.zeros(amps.shape, dtype=amps.dtype)
    ampsF[:,0] = filterAmpsSingle(amps[:,0], perc)
    ampsF[:,1] = filterAmpsSingle(amps[:,1], perc)
    return ampsF

def filterAmpsSingle(amps, perc):
    ampsF = np.zeros(amps.shape, dtype=amps.dtype)
    thrsh = np.max(np.abs(amps)) * perc
    for r, val in enumerate(amps):
        if np.abs(val) > thrsh:
            ampsF[r] = val
    return ampsF


def alter(amps):
    ampsF = filterAmps(amps, 0.5)
    ampsNew = addOctave(ampsF)
    return ampsNew


def plotSamples(samples):
    fig = plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(samples[:,0], samples[:,1])
    plt.draw()
    
def plotAmps(amps):
    N = len(amps)
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    ax0.plot(np.fft.fftfreq(N, d=delta), np.abs(amps[:,0]))
    ax1 = fig.add_subplot(122)
    ax1.plot(np.fft.fftfreq(N, d=delta), np.abs(amps[:,1]))
    plt.draw()

    

steps = 1000.0
target = 2*360.0
delta = target / steps
ts = np.linspace(0, target, steps)
sample = getSample(ts)
plotSamples(sample)
amps = fft(sample)
plotAmps(amps)
ampsNew = alter(amps)
plotAmps(ampsNew)
sampleNew = ifft(ampsNew)
plotSamples(sampleNew)
plt.show()




