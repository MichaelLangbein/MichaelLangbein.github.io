from numpy import zeros, exp, pi, complex128, shape, fft, abs, sin, cos, arange, log, linspace, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


spacing = 0.1
rx = 1
ry = 2
rz = 3
N = 20
M = 2*N


thetas = linspace(0, pi, N)
phis = linspace(0, 2*pi, M)
signal = zeros((N, M))
signalC = zeros((N * M, 3))
signalD = zeros((N * M, 3))
i = 0
for n, theta in enumerate(thetas):
    for m, phi in enumerate(phis):
        r = ellipsoid(theta, phi)
        x, y, z = sphericalToCart(theta, phi, r)
        signal[n][m] = r
        signalD[i] = [x, y, z]
        i += 1
        rcalc = sqrt(x**2 + y**2 + z**2 )
        print " r : {}  \n rc: {}".format(r, rcalc)






fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(signalD[:, 0], signalD[:, 1], signalD[:, 2], color="red")
ax.set_xlabel("x")
ax.set_xlabel("y")
ax.set_xlabel("z")
plt.show()
