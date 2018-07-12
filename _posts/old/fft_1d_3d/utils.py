import numpy as np
import matplotlib.pyplot as plt


def ellipse(a, b, theta):
    x = a * np.sin(theta)
    y = b * np.cos(theta)
    return x, y


def createData(a, b, N):
    thetas = np.linspace(0, 2 * np.pi, N)
    data  = np.zeros([100, 2])
    for r, theta in enumerate(thetas):
        data[r] = ellipse(a, b, theta)
    return data


def multiplot(pictures):
    fig = plt.figure()
    for el in pictures:
        kind = el[0]
        indx = el[1]
        data = el[2]
        axes = fig.add_subplot(indx)
        if kind == "scatter":
            doScatter(axes, data)
        elif kind == "plot":
            doPlot(axes, data)
    plt.show()

def doScatter(axes, data):
    for point in data:
        axes.scatter(point[0], point[1])

def doPlot(axes, data):
    axes.plot(data)
