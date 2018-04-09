import numpy as np


def fourier(data):
    return np.fft.fft(data)

def restore(amps):
    return np.fft.ifft(amps)

def highPassFilter(amps, thresh):
    newAmps = np.zeros(np.shape(amps), dtype=np.complex128)
    for r, val in enumerate(amps):
        if val > thresh:
            newAmps[r] = val
    return newAmps

def toRT(data):
    R = len(data)
    radii = np.zeros([R])
    for row, point in enumerate(data):
        radius = np.sqrt( point[0]**2 + point[1]**2 )
        radii[row] = radius
    thetas = np.linspace(0, 2.0 * np.pi, R)
    return radii, thetas

def toXY(radii, thetas):
    R = len(radii)
    XY = np.zeros([R, 2])
    for row in range(R):
        r = radii[row]
        t = thetas[row]
        x = r * np.cos(t)
        y = r * np.sin(t)
        XY[row,:] = [x,y]
    return XY
