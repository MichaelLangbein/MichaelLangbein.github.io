---
layout: post
title:  "Fourier Transform on Three-Dimensional Objects"
date:   2018-02-25 20:09:02 +0100
categories: fourier
---
$$ \newcommand{\Reals}[]{\mathbb{R}} $$
$$ \newcommand{\Naturals}[]{\mathbb{N}} $$
# Motivation: adding an octave to an ellipse

Like many people I first got into working with the Fourier transform when learning to work with music. It was neat how you could take a signal, find out what notes it consisted of and then adjust the signal to your liking by doing simple operations in the Fourier domain. For example you could add an octave on top of the most important note by just adding another frequency to the signal.

Really, a musical signal is only just a function of time, or, in more technical terms, a mapping $\Reals \to \Reals $. However, there is no reason why we should limit ourselves to only such functions. What about mappings of the form $ \Reals^n \to \Reals $, or $ \Reals^n \to \Reals^m $? It turns out that there is a way to decompose these functions into frequencies as well!

This opens up a whole new world of interesting transformations. For example, grayscale pictures can be viewed as a function mapping x- and y-coordinates onto a pixel-value between 0 and 255. This kind of function can be represented in the frequency domain.

Geometrical objects make for another interesting example. We can transform any geometrical object that has a parameterized equation. Consider for example the ellipse $ \\{ \vec{v} \| \frac{x^2}{r_1} + \frac{y^2}{r_2} + \frac{z^2}{r_3} = 1 \\} $. It has the parameterized form: 


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

Here, the radii $ r_1, r_2, r_3 $ are constants, and the parameters $ \theta, \phi $ vary between $ 0 $ and $ 2 \pi $.That means we can express an ellipse as a vector-valued function of the form $ [0, 2 \pi]^2 \to \Reals^3 $. Again, this function can be projected into the frequency domain.

What I initially found fascinating about the Fourier transform was how you could modify musical signals. Now that we have learned that you can just as well apply the Fourier transformation to geometrical objects, let's find out what it means to modify them in the Fourier domain! Is there such a thing as "adding an octave to a spere"? If so, what would the spere look like?


# A little bit of theory

Let's first back up a little bit. What does it actually mean to get the Fourier representation of a signal?

### Decomposing signals

Consider an inner product space $S$ with orthogonal base $B_S$. Any $\vec{s} \in S$ can be represented as a combination of the basevectors $\vec{b}_n \in B_S$, such that

$$\vec{s} = \sum \alpha_n \vec{b}_n $$

Since $B_S$ is orthogonal, the coefficients $\alpha_n$ are easy to obtain:

$$ \alpha_n = < \vec{s}, \vec{b_n} > $$

All this holds for any orthogonal base. Then how is a Fourier base special? Really, there isn't all that much special about the Fourier base, except that the coefficients $\alpha_n$ have a neat interpretation: they are the amplitdudes of a wave with frequency $f_n$. Chosing to represent a signal by it's Fourier coefficients is just like chosing to ............

### Different Fourier bases for different spaces

So when we do a Fourier transformation, all we do is find out the values of the amplitudes $\alpha_n$. Now it is important to understand that for different kinds of vectors there are different kinds of Fourier bases. Usually one first gets introduced to the space of periodic signals $\\{ \vec{s}_x \| ... \\}$. In this space, the Fourier base vectors have the form:

$$ ... $$

And the frequency associated with the amplitude $\alpha_n$ is

$$ f_n = ... $$

Here is a plot of some of these base functions.

...

The next interesting case is the space of signals $\\{ \vec{s}_{x, y} \| ... \\}$. Here, we can use the base vectors

$$ ... $$

The frequency associated with ....
Finally, 


# Working with the Fourier transform of geometrical objects

Now that we have some understanding of what we're doing, it's time to get out hands dirty. We'll transform an ellipse into the frequency domain, play around with the frequencies a bit, and transform it back to see what effect our meddling has had. 



```python
import numpy as np

def ellipse(theta, phi, r1, r2, r3):
    x = r1 * np.cos(theta) * np.cos(phi)
    y = r2 * np.cos(theta) * np.sin(phi)
    z = r3 * np.sin(theta)
    return x, y, z


r1 = r2 = r3 = 1
signal = np.zeros((360, 360, 3), dtype=np.float)
for theta in np.arange(0, 360):
    for phi in np.arange(0, 360):
    	data[theta, phi] = ellipse(theta, phi, r1, r2, r3)

```