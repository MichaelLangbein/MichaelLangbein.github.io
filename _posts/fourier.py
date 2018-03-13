from numpy import zeros, exp, pi, complex128, shape, fft, abs, sin, cos, arange, log, linspace, sqrt
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
    return fft.rfft2(signal)

def fourierTransformInverse(amps):
    return fft.irfft2(amps)


def getMax(mtrx):
    mx = 0
    N, M = shape(mtrx)
    for n in range(N):
        for m in range(M):
            if abs(mtrx[n][m]) > mx:
                mx = mtrx[n][m]
    return mx


def matixFilter(dataIn, filterFunc):
    N, M = shape(dataIn)
    dataOut = zeros((N, M), dtype=complex128)
    for n in range(N):
        for m in range(M):
            if filterFunc(n, m, dataIn[n][m]):
                dataOut[n][m] = dataIn[n][m]
    return dataOut

def matrixMap(matrix, mapFunc):
    N, M = shape(matrix)
    matrixNew = zeros((N, M), dtype=matrix.dtype)
    for n in range(N):
        for m in range(M):
            matrixNew[n][m] = mapFunc(n, m, matrix[n][m])
    return matrixNew


def ellipsoid(theta, phi):
    bx = cos(pi/2.0 - theta) * cos(phi) / rx
    by = cos(pi/2.0 - theta) * sin(phi) / ry
    bz = cos(theta) / rz
    r = sqrt( 1.0 / ( bx**2.0 + by**2.0 + bz**2.0 ) )
    return  r


def sphericalToCart(theta, phi, r):
    x = r * cos(pi/2.0 - theta) * cos(phi)
    y = r * cos(pi/2.0 - theta) * sin(phi)
    z = r * cos(theta)
    return x, y, z
            
def signalToCart(sig):
    N, M = shape(sig)
    signalCart = zeros((N * M, 3))
    i = 0
    for n,theta in enumerate(thetas):
        for m, phi in enumerate(phis):
            r = sig[n][m]
            cartCords = sphericalToCart(theta, phi, r)
            signalCart[i] = cartCords
            i += 1
    return signalCart  

# Step 0: constants
rx = 1
ry = 2
rz = 3
N = 50
M = 2*N

# Step 1: create data
print "Creating data"
thetas = linspace(0, pi, N)
phis = linspace(0, 2*pi, M)
signal = zeros((N, M))
for n,theta in enumerate(thetas):
    for m, phi in enumerate(phis):
        signal[n][m] = ellipsoid(theta, phi)



# Step 2: transform
print "Doing Fourier Transform"
amps = fourierTransform(signal)


# Step 3: manipulate
print "Manipulating data"
thrsh = getMax(amps) * 0.0
amps[0][0] = 0
#ampsNew = matixFilter(amps, lambda n, m, val : val > thrsh)
#ampsNew = matixFilter(amps, lambda n, m, val : (n - N/2)**2 + (m - M/4)**2 < 3*N )
#ampsNew = matixFilter(amps, lambda n, m, val : -30 < (n - N/2) - (m - M/4) < 30 )
#ampsNew = matrixMap(amps, lambda n, m, val : 4 * val)
ampsNew = matrixMap(amps, lambda n, m, val : ((n+1)%(m+1)) * val)

# Step 4: backtransform
print "Backtransform"
signalNew = fourierTransformInverse(ampsNew)






# Step 5: plot

signalC = signalToCart(signal)
signalNewC = signalToCart(signalNew)


def on_move(event):
    if event.inaxes == ax1:
        if ax1.button_pressed in ax1._rotate_btn:
            ax4.view_init(elev=ax1.elev, azim=ax1.azim)
        elif ax1.button_pressed in ax1._zoom_btn:
            ax4.set_xlim3d(ax1.get_xlim3d())
            ax4.set_ylim3d(ax1.get_ylim3d())
            ax4.set_zlim3d(ax1.get_zlim3d())
    elif event.inaxes == ax4:
        if ax4.button_pressed in ax4._rotate_btn:
            ax1.view_init(elev=ax4.elev, azim=ax4.azim)
        elif ax4.button_pressed in ax4._zoom_btn:
            ax1.set_xlim3d(ax4.get_xlim3d())
            ax1.set_ylim3d(ax4.get_ylim3d())
            ax1.set_zlim3d(ax4.get_zlim3d())
    else:
        return
    fig.canvas.draw_idle()


fig = plt.figure()
c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

ax1 = fig.add_subplot(231, projection='3d')
ax1.set_title("Original body")
ax1.set_aspect('equal', adjustable='box')
ax1.scatter(signalC[:, 0], signalC[:, 1], signalC[:, 2], cmap="viridis")

ax2 = fig.add_subplot(232)
ax2.set_title("Flattened")
ax2.imshow(signal)

ax3 = fig.add_subplot(233)
ax3.set_title("Fourier Transform")
ax3.imshow(log(abs(amps)+0.0001))

ax4 = fig.add_subplot(234, projection='3d')
ax4.set_title("New body")
ax4.set_aspect('equal', adjustable='box')
ax4.scatter(signalNewC[:, 0], signalNewC[:, 1], signalNewC[:, 2], cmap="viridis")

ax5 = fig.add_subplot(235)
ax5.set_title("Flattened")
ax5.imshow(abs(signalNew))

ax6 = fig.add_subplot(236)
ax6.set_title("Altered amplitudes")
ax6.imshow(log(abs(ampsNew)+0.0001))

plt.show()
