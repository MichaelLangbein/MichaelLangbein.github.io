import numpy as np

def ellipse(theta, phi, r1, r2, r3):
    x = r1 * np.cos(theta) * np.cos(phi)
    y = r2 * np.cos(theta) * np.sin(phi)
    z = r3 * np.sin(theta)
    return x, y, z


