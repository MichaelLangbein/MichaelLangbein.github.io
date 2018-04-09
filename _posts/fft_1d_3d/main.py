import numpy as np
import matplotlib.pyplot as plt
import utils as u
import fourierOneD as f1
import fourierTwoD as f2


data = u.createData(2, 3, 100)

radii, thetas = f1.toRT(data)
amps1 = f1.fourier(radii)
ampsAltered1 = f1.highPassFilter(amps1, None)
radiiNew1 = f1.restore(ampsAltered1)
dataNew1 = f1.toXY(radii, thetas)

amps2 = f2.fourier(data)
ampsAltered2 = f2.highPassFilter(amps2, None)
dataNew2 = f2.restore(ampsAltered2)

u.multiplot([
    ["scatter", "241", data], ["plot", "242", amps1], ["plot", "243", ampsAltered1], ["scatter", "244", dataNew1],
    ["scatter", "245", data], ["plot", "246", amps2], ["plot", "247", ampsAltered2], ["scatter", "248", dataNew2],
])
