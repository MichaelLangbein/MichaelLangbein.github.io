import numpy as np

def fourier(data):
    R = len(data)
    amps = np.zeros([R, 2], dtype=np.complex128)
    amps[:,0] = np.fft.fft(data[:,0])
    amps[:,1] = np.fft.fft(data[:,1])
    return amps

def restore(amps):
    R = len(amps)
    data = np.zeros([R,2])
    data[:,0] = np.fft.ifft(amps[:,0])
    data[:,1] = np.fft.ifft(amps[:,1])
    return data

def power(point):
    return np.sqrt(point[0]**2 + point[1**2])

def highPassFilter(amps, thresh):
    ampsNew = np.zeros(np.shape(amps), dtype=np.complex128)
    for r, point in enumerate(amps):
        p = power(point)
        print "Point: {} {} has power {}".format(point[0],point[1],p)
        if p > thresh:
            ampsNew[r] = point
    return ampsNew
