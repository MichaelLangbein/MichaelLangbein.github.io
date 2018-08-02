import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

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

def ifft(amps):
    samples = np.zeros(np.shape(amps))
    samples[:,:,0] = np.fft.ifft2(amps[:,:,0])
    samples[:,:,1] = np.fft.ifft2(amps[:,:,1])
    samples[:,:,2] = np.fft.ifft2(amps[:,:,2])
    return np.real(samples)

def matrixMap(mapFunc, matrix):
    matrixNew = np.zeros(matrix.shape, dtype=matrix.dtype)
    for r,row in enumerate(matrix):
        for c,el in enumerate(row):
            matrixNew[r,c] = mapFunc(r,c,el)
    return matrixNew


def matrixFilter(filterFunc, matrix):
    matrixNew = np.zeros(matrix.shape, dtype=matrix.dtype)
    for r,row in enumerate(matrix):
        for c,el in enumerate(row):
            if filterFunc(r,c,el):
                matrixNew[r,c] = matrix[r,c]
    return matrixNew


def addOctave(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,:,0] = addOctaveSingle(amps[:,:,0])
    ampsNew[:,:,1] = addOctaveSingle(amps[:,:,1])
    ampsNew[:,:,2] = addOctaveSingle(amps[:,:,2])
    return ampsNew


def addOctaveSingle(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    R,C = ampsNew.shape
    frqR = np.fft.fftfreq(R, d=delta)
    frqC = np.fft.fftfreq(C, d=delta)
    for r in range(R):
        for c in range(C):
            fr = frqR[r]
            fc = frqC[c]
            r2 = indexOf(2*fr, frqR)
            c2 = indexOf(2*fc, frqC)
            ampsNew[r,c] += amps[r,c]
            #ampsNew[r ,c2] += amps[r,c]
            #ampsNew[r2,c ] += amps[r,c]
            if r2 is not None and c2 is not None:
                fr2 = frqR[r2]
                fc2 = frqC[c2]
                #print("found double of {}:{}/{}:{} at {}:{}/{}:{}".format(r,fr,c,fc,r2,fr2,c2,fc2))
                ampsNew[r2,c2] += amps[r,c]
    return ampsNew

def indexOf(val, arr, maxDist=0.001, minDist=9999999999):
    indx = 0
    for i, el in enumerate(arr):
        dist = abs(val - el)
        if dist < minDist:
            indx = i
            minDist = dist
    if minDist < maxDist:
        return indx
    return None
    

def filterAmps(amps, perc):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,:,0] = filterAmpsSingle(amps[:,:,0], perc)
    ampsNew[:,:,1] = filterAmpsSingle(amps[:,:,1], perc)
    ampsNew[:,:,2] = filterAmpsSingle(amps[:,:,2], perc)
    return ampsNew

def filterAmpsSingle(amps, perc):
    thresh = np.max(np.abs(amps)) * perc
    ampsNew = matrixFilter(lambda r,c,val: np.abs(val) > thresh, amps)
    return ampsNew

def alter(amps):
    ampsF = filterAmps(amps, 0.9)
    ampsNew = addOctave(ampsF)
    return ampsNew


def plotSamples(samples):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for row in samples:
        for item in row:
            if item[0] > 0 and item[1] > 0 and item[2] > 0:
                ax.scatter(item[0],item[1],item[2])
    plt.draw()
    
def plotAmps(amps):
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.imshow(np.abs(amps[:,:,0]))
    ax1 = fig.add_subplot(132)
    ax1.imshow(np.abs(amps[:,:,1]))
    ax2 = fig.add_subplot(133)
    ax2.imshow(np.abs(amps[:,:,2]))
    plt.draw()

    

steps = 40.0
target = 3*360.0
delta = target / steps
thetas = np.linspace(0, target, steps)
phis = np.linspace(0, target, steps)
sample = getSample(thetas, phis)
plotSamples(sample)
amps = fft(sample)
plotAmps(amps)
ampsNew = alter(amps)
plotAmps(ampsNew)
sampleNew = ifft(ampsNew)
plotSamples(sampleNew)
plt.show()




