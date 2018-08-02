import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

r1 = 2
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
    samples = np.zeros(np.shape(amps))
    samples[:,0] = np.fft.ifft(amps[:,0])
    samples[:,1] = np.fft.ifft(amps[:,1])
    return samples

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


class Circle:
    
    def __init__(self, n, initial):
        self.length = n
        self.data = np.zeros(n, dtype=object)
        for i in range(self.length):
            self.data[i] = initial
    
    def insertAt(self, index, value):
        for i in range(self.length-1, index-1, -1):
            self.data[i] = self.data[i-1]
        self.data[index] = value

    def insertWhere(self, func, value):
        for i in range(self.length):
            if func(self.data[i]):
                self.insertAt(i, value)
                break

class Entry:

    def __init__(self, r, c, val):
        self.row = r
        self.col = c
        self.val = val

    def __str__(self):
        return "[{},{}:{}]".format(self.row, self.col, self.val)

    def __repr__(self):
        return self.__str__()


def getLargest(data, n):
    largest = Circle(n, Entry(-1,-1,-9999999999))
    for r,row in enumerate(data):
        for c,val in enumerate(row):
            entry = Entry(r,c,val)
            largest.insertWhere((lambda other : entry.val > other.val),  entry)
    return largest.data


def getLargestAmps(amps, n):
    lengths = amps[:,:,0]*amps[:,:,0] + amps[:,:,1]*amps[:,:,1] + amps[:,:,2]*amps[:,:,2] 
    largest = getLargest(lengths, n)
    for l in largest:
        l.val = amps[l.row, l.col]
    return largest


def indexOf(val, arr):
    indx = 0
    minDist = 9999999999
    for i, el in enumerate(arr):
        dist = abs(val - el)
        if dist < minDist:
            indx = i
            minDist = dist
    return indx


def addOctave(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,0] = addOctaveSingle(amps[:,0])
    ampsNew[:,1] = addOctaveSingle(amps[:,1])
    return ampsNew


def addOctaveSingle(amps):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    R = len(amps)
    frqR = np.fft.fftfreq(R, d=delta)
    for r in range(R):
            fr = frqR[r]
            r2 = indexOf(2*fr, frqR)
            ampsNew[r] += amps[r]
            ampsNew[r2] += amps[r]
    return ampsNew


def alter(amps):
    ampsNew = addOctave(amps)
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
    ax0.scatter(np.fft.fftfreq(N, d=delta), amps[:,0])
    ax1 = fig.add_subplot(122)
    ax1.scatter(np.fft.fftfreq(N, d=delta), amps[:,1])
    plt.draw()

    

steps = 1000.0
target = 2*360.0
delta = target / steps
ts = np.linspace(0, target, steps)
sample = getSample(ts)
print sample
plotSamples(sample)
amps = fft(sample)
plotAmps(amps)
ampsNew = alter(amps)
plotAmps(ampsNew)
sampleNew = ifft(ampsNew)
#sampleNew = getAnalyticalSollution(thetas, phis)
plotSamples(sampleNew)
plt.show()




