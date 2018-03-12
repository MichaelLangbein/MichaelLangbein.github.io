from numpy import zeros, exp, pi, complex128, shape, fft, abs, sin, cos, arange, log, linspace, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ellipsoid(theta, phi):
    theta2 = theta + pi/2.0
    x = rx * cos(theta2) * cos(phi)
    y = ry * cos(theta2) * sin(phi)
    z = rz * sin(theta2)
    r = sqrt(x**2 + y**2 + z**2)
    return x, y, z, r


def sphericalToCart(theta, phi, r):
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return [x, y, z]


spacing = 0.1
rx = 1
ry = 2
rz = 3
N = 50
M = 100


thetas = linspace(0, pi, N)
phis = linspace(0, 2*pi, M)
signal = zeros((N, M))
signalC = zeros((N * M, 3))
signalD = zeros((N * M, 3))
i = 0
for n, theta in enumerate(thetas):
    for m, phi in enumerate(phis):
        x, y, z, r = ellipsoid(theta, phi)
        signal[n][m] = r
        signalC[i] = [x, y, z]
        signalD[i] = sphericalToCart(theta, phi, r)
        i += 1






fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(signalC[:, 0], signalC[:, 1], signalC[:, 2], color="blue")
ax.scatter(signalD[:, 0], signalD[:, 1], signalD[:, 2], color="red")
ax.set_xlabel("x")
ax.set_xlabel("y")
ax.set_xlabel("z")
plt.show()
