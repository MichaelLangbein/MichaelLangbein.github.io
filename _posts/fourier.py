from numpy import zeros, exp, pi, complex128, shape, fft, abs, sin, cos, arange, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def amplitudeFunc(fr1, fr2, signal):
    X, Y = shape(signal)
    sm = 0
    for x in range(X):
        for y in range(Y):
            sm += signal[x][y] * exp( -1j * 2.0 * pi * ( x*fr1/X + y*fr2/Y ) )
    return sm


def myfourierTransform(signal):
    N, M = shape(signal)
    amps = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            amps[n][m] = amplitudeFunc(n, m, signal)
    return amps


def myfourierTransformInverse(amps):
    N, M = shape(amps)
    signal = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            signal[n][m] = 1.0 / amplitudeFunc(n, m, amps)
    return signal


def fourierTransform(signal):
    return fft.fft2(signal)

def fourierTransformInverse(amps):
    return fft.ifft2(amps)


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


def ellipsoid(theta, phi):
    br1 = cos(theta)**2.0 * sin(phi)**2.0 / rx**2.0
    br2 = sin(theta)**2.0 * sin(phi)**2.0 / ry**2.0
    br3 = cos(phi)**2.0 / rz**2.0
    r = (1.0 / (br1 + br2 + br3) )**(0.5)
    return r
            
    

# Step 0: constants
rx = 2
ry = 3
rz = 1
spacing = 0.2

# Step 1: create data
print "Creating data"
signal = []
for theta in arange(0.0, 2.0*pi, spacing):
    row = []
    for phi in arange(0.0, 2.0*pi, spacing):
        row.append( ellipsoid(theta, phi) )
    signal.append(row)


# Step 2: transform
print "Doing Fourier Transform"
amps = fourierTransform(signal)


# Step 3: manipulate
print "Manipulating data"
thrsh = getMax(amps) * 0
amps[0][0] = 0
ampsNew = filterBy(amps, lambda val : val > thrsh)


# Step 4: backtransform
print "Backtransform"
signalNew = fourierTransformInverse(ampsNew)






# Step 5: plot

def sphericalToCart(theta, phi, r):
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r  * cos(phi)
    return [x, y, z]


def signalToCart(sig):
    N, M = shape(sig)
    signalCart = []
    for n in range(N):
        row = []
        for m in range(M):
            theta = spacing * n
            phi = spacing * m
            r = sig[n][m]
            cartCords = sphericalToCart(theta, phi, r)
            row.append(cartCords)
        signalCart.append(row)
    return signalCart

signalC = signalToCart(signal)
signalNewC = signalToCart(signalNew)


fig = plt.figure()

ax1 = fig.add_subplot(231, projection='3d')
ax1.set_title("Original body")
for point in signalC:
    ax1.scatter(point[0], point[1], point[2])

ax2 = fig.add_subplot(232)
ax2.set_title("Flattened")
ax2.imshow(signal)

ax3 = fig.add_subplot(233)
ax3.set_title("Fourier Transform")
ax3.imshow(log(abs(amps)))

ax4 = fig.add_subplot(234, projection='3d')
ax4.set_title("New body")
for point in signalNewC:
    ax4.scatter(point[0], point[1], point[2])

ax5 = fig.add_subplot(235)
ax5.set_title("Flattened")
ax5.imshow(abs(signalNew))

ax6 = fig.add_subplot(236)
ax6.set_title("Altered amplitudes")
ax6.imshow(log(abs(ampsNew)))

plt.show()
