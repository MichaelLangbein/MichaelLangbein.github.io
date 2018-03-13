---
layout: post
title:  "Fourier Transform on Three-Dimensional Objects: Part 1"
date:   2018-02-25 20:09:02 +0100
categories: fourier
---

This post is still in draft stage! I'll finish it up as soon as I find some time to write again.

$$ \newcommand{\Reals}[]{\mathbb{R}} $$
$$ \newcommand{\Complexes}[]{\mathbb{C}} $$
$$ \newcommand{\Naturals}[]{\mathbb{N}} $$
# Motivation

Like many people I first got into working with the Fourier transform when learning to work with music. It was neat how you could take a signal, find out what notes it consisted of and then adjust the signal to your liking by doing simple operations in the Fourier domain. For example you could add an octave on top of the most important note by just adding another frequency to the signal.

Really, a musical signal is only just a function of time, or, in more technical terms, a mapping $\Reals \to \Reals $. However, there is no reason why we should limit ourselves to only such functions. What about mappings of the form $ \Reals^n \to \Reals $, or $ \Reals^n \to \Reals^m $? It turns out that there is a way to decompose these functions into frequencies as well!

This opens up a whole new world of interesting transformations. For example, grayscale pictures can be viewed as a function mapping x- and y-coordinates onto a pixel-value between 0 and 255. Geometrical objects make for another interesting example. In this post, we'll work with an geometrical object that can be described by an explicit function of the form $\Reals^2 \to \Reals$, the ellipsoid. But there is no reason why we could not work with any other  function from $L^2(\Complexes^n \to \Complexes)$. In the next post, we'll even go one step further and deal with vector valued functions from $L^2(\Complexes^n \to \Complexes^m)$.

What I initially found fascinating about the Fourier transform was how you could modify musical signals. Now that we have learned that you can just as well apply the Fourier transformation to geometrical objects, let's find out what it means to modify them in the Fourier domain! Is there such a thing as "adding an octave to a spere"? If so, what would the spere look like?


# A little bit of theory

Let's first back up a little bit. What does it actually mean to get the Fourier representation of a signal?

### Decomposing signals

Consider an inner product space $S$ with orthogonal base $B_S$. Any $\vec{s} \in S$ can be represented as a combination of the basevectors $\vec{b}_n \in B_S$, such that

$$\vec{s} = \sum \alpha_n \vec{b}_n $$

Since $B_S$ is orthogonal, the coefficients $\alpha_n$ are easy to obtain:

$$ \alpha_n = < \vec{s}, \vec{b_n} > $$

All this holds for any orthogonal base. Then how is a Fourier base special? Really, there isn't all that much special about the Fourier base, except that the coefficients $\alpha_n$ have a neat interpretation: they are the amplitudes of a wave with frequency $f_n$.

The Fourier Transform is a transform on $B$ that changes the base of $B$ to a set of functions that can be viewed as sine waves. If the signal we want to represent does indeed have some wave-like properties, this representation may be a lot more concise. Choosing to represent a signal by it's Fourier coefficients is a bit like choosing to describe a piece of music by it's notesheet instead of, say, a wav file. The information content remains the same, but from the notesheet we can extract some information way more easily from a notesheet than from the raw values of the wav file. Most people will be able to get at least a vague impression of a song from it's notes, but very few will be able to read *anything* out of the long list of speaker-positions that make out a wav file.


<!--
### Different Fourier bases for different spaces

So when we do a Fourier transformation, all we do is find out the values of the amplitudes $\alpha_n$. Now it is important to understand that for different kinds of vectors there are different kinds of Fourier bases. Usually one first gets introduced to the space of periodic signals $\\{ \vec{s}_x \| ... \\}$. In this space, the Fourier base vectors have the form:

$$ ... $$

And the frequency associated with the amplitude $\alpha_n$ is

$$ f_n = ... $$

Here is a plot of some of these base functions.

...

The next interesting case is the space of signals $\\{ \vec{s}_{x, y} \| ... \\}$. In the example of the grayscale-image from earlier, we mapped coordinates $x, y$ to a grayscale-value. Here, we can use the base vectors

$$ ... $$

The frequency associated with ....
Finally, lets revisit the example of the ellipsoid-function from the introduction. Contrary to the case of the grayscale image, where we mapped coordinates to a value, now we'll map parameters to coordinates. Consequently, our base will consist of functions mapping .... .

This may be confusing in a way: both in the case of the grayscale image and in the case of the ellipsoid our Fourier amplitudes have only two dimensions! How is this possible when an image is clearly two-dimensional, whereas an ellipsoid has three dimensions? Well, what determines the dimensionality of the Fourier amplitudes is the *domain* of the function, not its range. In the case of the image, the x- and y-coordinates are important for the dimensionality of the amplitudes, not the grayscale-value. In the case of the ellipsoid, the parameters $\theta$ and $\phi$ matter for the dimensionality of the amplitudes, not the coordinates that the ellipsoid-function yields. It just so happend that the ellipsoid can be parameterized with two parameters, but there are other geometric objects that require less, or more, parameters. These functions, however, would make out part of different vector spaces and as such have their own Fourier bases.  
-->

# Working with the Fourier transform of geometrical objects

Now that we have some understanding of what we're doing, it's time to get out hands dirty. We'll transform an ellipsoid into the frequency domain, play around with the frequencies a bit, and transform it back to see what effect our meddling has had.

First, however, we'll need a function to describe the ellipsoid. For convenience, we'll use a spherical coordinate system. Within such a system, we can write the radius of the ellipsoid as a function of the two parameters $\theta$ (the polar angle) and $\phi$ (the azimuth angle):

$$ r = 1 / \sqrt{ \frac{\cos^2( \frac{\pi}{2} - \theta) \cos^2(\phi)}{r_x^2} + \frac{\cos^2( \frac{\pi}{2} -\theta) \sin^2(\phi)}{r_y^2} + \frac{\cos^2(\theta)}{r_z^2} } $$

Let's implement this:



```python
from numpy import zeros, exp, pi, complex128, shape, fft, abs, sin, cos, arange, log, linspace, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fourierTransform(signal):
    return fft.rfft2(signal)

def fourierTransformInverse(amps):
    return fft.irfft2(amps)

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


# Step 0: constants
rx = 1
ry = 4
rz = 1
N = 30
M = 2*N


# Step 1: create data
thetas = linspace(0, pi, N)
phis = linspace(0, 2*pi, M)
signal = zeros((N, M))
for n,theta in enumerate(thetas):
    for m, phi in enumerate(phis):
        signal[n][m] = ellipsoid(theta, phi)


# Step 2: transform
amps = fourierTransform(signal)


# Step 3: manipulate
ampsNew = matixFilter(amps, lambda n, m, val : val > 0)


# Step 4: backtransform
signalNew = fourierTransformInverse(ampsNew)

```

This results in a ........



How about if we use this manipulation instead?
```python
ampsNew = matixFilter(amps, lambda n, m, val : -N/4 < (n - N/2) - (m - M/4) < N/4 )
```


# Conclusion ... for now

So now we have seen how we may take any (square-integrable) function and play around with it in the frequency domain. But there was a little bit of cheating involved. By restricting our attention to only cases where an explicit function of the form $\Reals^n \to \Reals$, we conveniently blended out a huge amount of problematic cases, most notably, geometrical objects that cannot be written in this explicit form. In fact, even the example of the ellipsoid that we have seen here can be too complex for this simplified approach. For simplicity, we decided to express the ellipsoid inside a spherical coordinate system, which allowed us to find an explicit expression describing the ellipsoid as $r = f(\theta, \phi)$. Had we chosen a cartesian coordinate system instead, no such function would exist! We'd be dealing with a function of the form

$$
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
=
\begin{bmatrix}
r_1 \cos \theta \sin \phi \\
r_2 \sin \theta \sin \phi \\
r_3 \cos \theta
\end{bmatrix}
$$

That is, we'd be dealing with vector valued functions. When you want to work with arbitrary geometric objects, you'll probably come across this case way more often than the few instances where you can find a nice, explicit function. So in the next post of this series, we'll deal with that case! Stay tuned!
