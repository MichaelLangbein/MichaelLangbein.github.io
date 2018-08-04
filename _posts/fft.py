import numpy as np
import mayavi.mlab as mm


r1 = r2 = r3 = 1
def body(theta, phi):
    x = r1 * np.cos(theta) * np.cos(phi)
    y = r2 * np.cos(theta) * np.sin(phi)
    z = r3 * np.sin(theta)
    return [x, y, z]

def bodyOctave(theta, phi):
    x = r1 * np.cos(theta) * np.cos(phi) + r1 * np.cos(2*theta) * np.cos(2*phi)
    y = r2 * np.cos(theta) * np.sin(phi) + r2 * np.cos(2*theta) * np.sin(2*phi)
    z = r3 * np.sin(theta) + r3 * np.sin(2*theta)
    return [x, y, z]


def newBody(theta, phi):
    x = r1 * np.sin(theta) - np.cos(phi)
    y = r2 * np.sin(theta) * np.cos(phi)
    z = r3 * np.cos(theta) - np.sin(phi)
    return [x, y, z]


def getSample(thetas, phis, body):
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
    samples = np.zeros(np.shape(amps), dtype=amps.dtype)
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
            if r < R/4.0:
                r2 = 2*r
            elif r > 3.0*R/4.0:
                r2 = 2*r - R
            else:
                r2 = None
            if c < C/4.0:
                c2 = 2*c
            elif c > 3.0*C/4.0:
                c2 = 2*c - C
            else:
                c2 = None
            ampsNew[r,c] += amps[r,c]
            if r2 is not None and c2 is not None:
                ampsNew[r2,c2] += amps[r,c]
                if amps[r,c] != 0.0:
                    print("duplicating value {} from {}/{} to {}/{}".format(amps[r,c], r,c, r2,c2))
    return ampsNew

    

def filterAmps(amps, perc):
    ampsNew = np.zeros(amps.shape, dtype=amps.dtype)
    ampsNew[:,:,0] = filterAmpsSingle(amps[:,:,0], perc)
    ampsNew[:,:,1] = filterAmpsSingle(amps[:,:,1], perc)
    ampsNew[:,:,2] = filterAmpsSingle(amps[:,:,2], perc)
    return ampsNew

def filterAmpsSingle(amps, perc):
    thresh = np.max(np.abs(amps)) * perc
    def filterfunc(r, c, val):
        if np.abs(val) > thresh:
            print("retaining {} at {}/{}".format(val, r, c))
            return True
        return False
    ampsNew = matrixFilter(filterfunc, amps)
    return ampsNew

def alter(amps):
    ampsF = filterAmps(amps, 0.96)
    ampsNew = addOctave(ampsF)
    return ampsNew


def plotSamples(samples, name):
    fig = mm.figure(name)
    x = samples[:,:,0]
    y = samples[:,:,1]
    z = samples[:,:,2]
    mm.points3d(x, y, z, figure=fig)
    
def plotAmps(amps, name):
    fig = mm.figure(name)
    mm.imshow(np.log(np.abs(amps[:,:,0])+0.000001), figure=fig)
    mm.imshow(np.log(np.abs(amps[:,:,1])+0.000001), figure=fig)
    mm.imshow(np.log(np.abs(amps[:,:,2])+0.000001), figure=fig)

    

steps = 250.0
target = 360.0
delta = target / steps
thetas = np.linspace(0, target, steps)
phis = np.linspace(0, target, steps)
sample = getSample(thetas, phis, newBody)
plotSamples(sample, "sample")
amps = fft(sample)
plotAmps(amps, "amps")
ampsNew = alter(amps)
plotAmps(ampsNew, "ampsNew")
sampleNew = ifft(ampsNew)
plotSamples(sampleNew, "sampleNew")
#samplesAnaly = getSample(thetas, phis, bodyOctave)
#lotSamples(samplesAnaly)

mm.show()
