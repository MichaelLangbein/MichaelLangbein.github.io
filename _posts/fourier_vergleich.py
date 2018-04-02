import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Step 1: create testdata
N = 30
M = 2 * N
thetas = np.linspace( - np.pi/2, np.pi/2, N)
phis   = np.linspace( - np.pi, np.pi, M)

a = 2
b = 3
c = 1
def ellipsoid(theta, phi):
    x = a * np.cos(theta) * np.cos(phi)
    y = b * np.cos(theta) * np.sin(phi)
    z = c * np.sin(theta)
    return x, y, z

dataset = np.zeros([N, M, 3])
for n, theta in enumerate(thetas):
    for m, phi in enumerate(phis):
        dataset[n,m,:] = ellipsoid(theta, phi)


# Step 2: do transform
transfDataset = np.zeros([N, M, 3], dtype=np.complex128)
transfDataset[:,:,0] = np.fft.fft(dataset[:,:,0])
transfDataset[:,:,1] = np.fft.fft(dataset[:,:,1])
transfDataset[:,:,2] = np.fft.fft(dataset[:,:,2])


# Step 3: mutating the data
def matrixApply(func, matrix, newMatrix):
    R, C = np.shape(matrix)
    for r in range(R):
        for c in range(C): 
            newMatrix[r,c] = func(r, c, matrix[r,c])

def center(r, c, val):
    if (r - N/2)**2 + (c - M/2)**2 > 6*N:
        return val
    return 0

def top(r, c, val):
    if r < N/2:
        return val
    return 0

def mult(r, c, val):
    return val/((r-N/2)**2 + (c-M/2)**2)

def cross(r, c, val):
    if abs(r - N/2) < 6 or abs(c - M/2) < 6:
        return val
    return 0


alteredDataset = np.zeros([N, M, 3], dtype=np.complex128)
matrixApply(cross, transfDataset[:,:,0], alteredDataset[:,:,0])
matrixApply(cross, transfDataset[:,:,1], alteredDataset[:,:,1])
matrixApply(cross, transfDataset[:,:,2], alteredDataset[:,:,2])


# Step 4: backtransform
newDataset = np.zeros([N, M, 3])
newDataset[:,:,0] = np.fft.ifft(alteredDataset[:,:,0])
newDataset[:,:,1] = np.fft.ifft(alteredDataset[:,:,1])
newDataset[:,:,2] = np.fft.ifft(alteredDataset[:,:,2])



# Step 5: plotting

def mtrxToArr(mtrx):
    R, C, D = np.shape(mtrx)
    out = np.zeros([R*C, D])
    i = 0
    for r in range(R):
        for c in range(C):
            out[i] = mtrx[r, c, :]
            i += 1
    return out
    

def on_move(event):
    if event.inaxes == ax1:
        if ax1.button_pressed in ax1._rotate_btn:
            ax5.view_init(elev=ax1.elev, azim=ax1.azim)
        elif ax1.button_pressed in ax1._zoom_btn:
            ax5.set_xlim3d(ax1.get_xlim3d())
            ax5.set_ylim3d(ax1.get_ylim3d())
            ax5.set_zlim3d(ax1.get_zlim3d())
    elif event.inaxes == ax5:
        if ax5.button_pressed in ax5._rotate_btn:
            ax1.view_init(elev=ax5.elev, azim=ax5.azim)
        elif ax5.button_pressed in ax5._zoom_btn:
            ax1.set_xlim3d(ax5.get_xlim3d())
            ax1.set_ylim3d(ax5.get_ylim3d())
            ax1.set_zlim3d(ax5.get_zlim3d())
    else:
        return
    fig.canvas.draw_idle()



fig = plt.figure()
c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)


datasetFlat = mtrxToArr(dataset)
ax1 = fig.add_subplot(241, projection='3d')
ax1.set_title("Original body")
ax1.set_aspect('equal', adjustable='box')
ax1.scatter(datasetFlat[:,0], datasetFlat[:,1], datasetFlat[:,2], cmap="viridis")

ax2 = fig.add_subplot(242)
ax2.set_title("Amplitides X")
ax2.imshow(np.abs(transfDataset[:,:,0]))

ax3 = fig.add_subplot(243)
ax3.set_title("Amplitides Y")
ax3.imshow(np.abs(transfDataset[:,:,1]))

ax4 = fig.add_subplot(244)
ax4.set_title("Amplitides Z")
ax4.imshow(np.abs(transfDataset[:,:,2]))






newDatasetFlat = mtrxToArr(newDataset)
ax5 = fig.add_subplot(245, projection='3d')
ax5.set_title("New body")
ax5.set_aspect('equal', adjustable='box')
ax5.scatter(newDatasetFlat[:,0], newDatasetFlat[:,1], newDatasetFlat[:,2], cmap="viridis")

ax6 = fig.add_subplot(246)
ax6.set_title("New Amplitides X")
ax6.imshow(np.abs(alteredDataset[:,:,0]))

ax7 = fig.add_subplot(247)
ax7.set_title("New Amplitides Y")
ax7.imshow(np.abs(alteredDataset[:,:,1]))

ax8 = fig.add_subplot(248)
ax8.set_title("New Amplitides Z")
ax8.imshow(np.abs(alteredDataset[:,:,2]))



plt.show()
