from numpy import zeros, exp, pi, complex128, shape, fft, abs
import matplotlib.pyplot as plt


def amplitudeFunc(fr1, fr2, signal):
    X, Y = shape(signal)
    sm = 0
    for x in range(X):
        for y in range(Y):
            sm += signal[x][y] * exp( -1j * 2.0 * pi * ( x*fr1/X + y*fr2/Y ) )
    return sm


def fourierTransform(signal):
    N, M = shape(signal)
    amps = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            amps[n][m] = amplitudeFunc(n, m, signal)
    return amps


def fourierTransformInverse(amps):
    N, M = shape(amps)
    signal = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            signal[n][m] = 1.0 / amplitudeFunc(n, m, amps)
    return signal


def getMax(mtrx):
    mx = 0
    N, M = shape(mtrx)
    for n in range(N):
        for m in range(M):
            if abs(mtrx[n][m]) > mx:
                mx = mtrx[n][m]
    return mx


def filterBy(dataIn, filterFunc):
    N, M = shape(signal)
    dataOut = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            if filterFunc(dataIn[n][m]):
                dataOut[n][m] = dataIn[n][m]
    return dataOut


def plotFour(s1, a1, s2, a2):
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(abs(s1))
    plt.subplot(222)
    plt.imshow(abs(a1))
    plt.subplot(223)
    plt.imshow(abs(s2))
    plt.subplot(224)
    plt.imshow(abs(a2))
    plt.show()


# Step 1: create data
signal = [[1,2],[3,4]]

# Step 2: transform
amps = fourierTransform(signal)

# Step 3: manipulate
thrsh = getMax(amps) * 0.95
ampsNew = filterBy(amps, lambda val : val > thrsh)

# Step 4: backtransform
signalNew = fourierTransformInverse(ampsNew)

# Step 5: plot
plotFour(signal, amps, signalNew, ampsNew)
