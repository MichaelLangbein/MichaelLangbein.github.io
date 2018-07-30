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

def ifft(amps):
    samples = np.zeros(np.shape(amps))
    samples[:,:,0] = np.fft.ifft2(amps[:,:,0])
    samples[:,:,1] = np.fft.ifft2(amps[:,:,1])
    samples[:,:,2] = np.fft.ifft2(amps[:,:,2])
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
    for r,val in enumerate(data):
        entry = Entry(r,None,val)
        largest.insertWhere((lambda other : entry.val > other.val),  entry)
    return largest.data


def getLargestM(data, n):
    largest = Circle(n, Entry(-1,-1,-9999999999))
    for r,row in enumerate(data):
        for c,val in enumerate(row):
            entry = Entry(r,c,val)
            largest.insertWhere((lambda other : entry.val > other.val),  entry)
    return largest.data


def getLargestAmps(amps, n):
    lengths = amps[:,:,0]*amps[:,:,0] + amps[:,:,1]*amps[:,:,1] + amps[:,:,2]*amps[:,:,2] 
    largest = getLargestM(lengths, n)
    for l in largest:
        l.val = amps[l.row, l.col]
    return largest

def getLargestAmpsBiggerOne(amps, n):
    lengths = amps[:,:,0]*amps[:,:,0] + amps[:,:,1]*amps[:,:,1] + amps[:,:,2]*amps[:,:,2] 
    largest = Circle(n, Entry(-1,-1,-9999999999))
    for r,row in enumerate(lengths):
        for c,val in enumerate(row):
            if r >= 1 and c >= 1:
                entry = Entry(r,c,val)
                largest.insertWhere((lambda other : entry.val > other.val),  entry)
    out = largest.data
    for l in out:
        l.val = amps[l.row, l.col]
    return out


def retainTopN(amps, n):
    largest = getLargestAmps(amps, n)
    largestIndices = list(map(lambda el : [el.row, el.col], largest))
    def indexInLargest(r,c,val):
        if [r,c] in largestIndices:
            print("Keeping {},{}={}".format(r,c,val))
            return val
        else:
            return [0,0,0]
    ampsNew = matrixMap(indexInLargest, amps)
    return ampsNew



def retainHighestLines(amps, nxr, nxc, nyr, nyc, nzr, nzc):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,:,0] = retainHighestLinesSingle(amps[:,:,0], nxr, nxc)
    ampsNew[:,:,1] = retainHighestLinesSingle(amps[:,:,1], nyr, nyc)
    ampsNew[:,:,2] = retainHighestLinesSingle(amps[:,:,2], nzr, nzc)
    return ampsNew

def retainHighestLinesSingle(amps, nRows, nCols):
    rowsCuml = amps.sum(axis=1)
    largestRows = getLargest(rowsCuml, nRows)
    colsCuml = amps.sum(axis=0)
    largestCols = getLargest(colsCuml, nCols)
    def simplify(r,c,val):
        for lr in largestRows:
            if r == lr.row:
                return lr.val
        for lc in largestCols:
            if c == lc.row:
                return lc.val
        return 0.0
    ampsNew = matrixMap(simplify, amps)
    return ampsNew


def addOctave(amps, n):
    largest = getLargestAmpsBiggerOne(amps, n)
    print("Adding octave to values {}".format(largest))
    def isMultipleOf(val1, val2):
        if val1 <= 2:
            return False
        if val2 <= 2:
            return False
        return val1 % val2 == 0
    def indexMultipleOfLargest(r,c,val):
        valOut = val
        for li in largest:
            if isMultipleOf(r, li.row) and isMultipleOf(c, li.col):
                print("{},{}={} is a multiple of {},{}={}".format(r,c,val,li.row,li.col,li.val))
                valOut += li.val
        return valOut
    ampsNew = matrixMap(indexMultipleOfLargest, amps)
    return ampsNew
        

def doubleLines(amps, n):
    numR, numC, numD = amps.shape
    largest = getLargestAmpsBiggerOne(amps, n)
    rows = list(map(lambda el: el.row, largest))
    cols = list(map(lambda el: el.col, largest))
    for row in rows:
        amps[2*row%numR, :, :] += amps[row, :, :]
    for col in cols:
        amps[:, 2*col%numC, :] += amps[:, col, :]
    return amps

def addMidLine(amps):
    numR, numC, numD = amps.shape
    largest = getLargestAmpsBiggerOne(amps, 1)
    r = largest[0].row
    c = largest[0].col
    amps[int(numR/2), :, :] = amps[r, :, :]
    amps[:, int(numC/2), :] = amps[:, c, :]
    return amps


def alter(amps):
    ampsFiltered = retainHighestLines(amps, 2,2, 2,2, 1,0)
    ampsNew = ampsFiltered # doubleLines(ampsFiltered, 2)
    return ampsNew


def plotSamples(samples):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for row in samples:
        for item in row:
            ax.scatter(item[0],item[1],item[2])
    plt.draw()
    
def plotAmps(amps):
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.imshow(np.log(np.abs(amps[:,:,0])+0.0001))
    ax1 = fig.add_subplot(132)
    ax1.imshow(np.log(np.abs(amps[:,:,1])+0.0001))
    ax2 = fig.add_subplot(133)
    ax2.imshow(np.log(np.abs(amps[:,:,2])+0.0001))
    plt.draw()
    

steps = 40.0
target = 3*360.0
delta = target / steps
thetas = np.linspace(0, target, steps)
phis = np.linspace(0, target, steps)
sample = getSample(thetas, phis)
plotSamples(sample)
plotAmps(sample)
amps = fft(sample)
plotAmps(amps)
ampsNew = alter(amps)
plotAmps(ampsNew)
sampleNew = ifft(ampsNew)
plotAmps(sampleNew)
plotSamples(sampleNew)
plt.show()




