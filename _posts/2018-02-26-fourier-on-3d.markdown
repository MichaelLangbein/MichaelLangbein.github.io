---
layout: post
title:  "Fourier Transform on Three-Dimensional Objects"
date:   2018-02-25 20:09:02 +0100
categories: fourier
---
$$ \newcommand{\Reals}[]{\mathbb{R}} $$
$$ \newcommand{\Naturals}[]{\mathbb{N}} $$
# Motivation

Like many people I first got into working with the Fourier transform when learning to work with music. It was neat how you could take a signal, find out what notes it consisted of and then adjust the signal to your liking by doing simple operations in the Fourier domain. For example you could add an octave on top of the most important note by just adding another frequency to the signal.

Really, a musical signal is only just a function of time, or, in more technical terms, a mapping $\Reals \to \Reals $. However, there is no reason why we should limit ourselves to only such functions. What about mappings of the form $ \Reals^n \to \Reals $, or $ \Reals^n \to \Reals^m $? It turns out that there is a way to decompose these functions into frequencies as well!

This opens up a whole new world of interesting transformations. For example, grayscale pictures can be viewed as a function mapping x- and y-coordinates onto a pixel-value between 0 and 255. Geometrical objects make for another interesting example. We can transform any geometrical object that has a parameterized equation. Consider for example the ellipsoid $ \\{ \vec{v} \| \frac{x^2}{r_1} + \frac{y^2}{r_2} + \frac{z^2}{r_3} = 1 \\} $. It has the parameterized form:


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

Here, we'll keep the radii $ r_1, r_2, r_3 $ as constants, and let the parameters $ \theta, \phi $ vary between $ 0 $ and $ 2 \pi $.That means we can express an ellipsoid as a vector-valued function of the form $ [0, 2 \pi]^2 \to \Reals^3 $. All these functions can be projected into the frequency domain.

What I initially found fascinating about the Fourier transform was how you could modify musical signals. Now that we have learned that you can just as well apply the Fourier transformation to geometrical objects, let's find out what it means to modify them in the Fourier domain! Is there such a thing as "adding an octave to a spere"? If so, what would the spere look like?


# A little bit of theory

Let's first back up a little bit. What does it actually mean to get the Fourier representation of a signal?

### Decomposing signals

Consider an inner product space $S$ with orthogonal base $B_S$. Any $\vec{s} \in S$ can be represented as a combination of the basevectors $\vec{b}_n \in B_S$, such that

$$\vec{s} = \sum \alpha_n \vec{b}_n $$

Since $B_S$ is orthogonal, the coefficients $\alpha_n$ are easy to obtain:

$$ \alpha_n = < \vec{s}, \vec{b_n} > $$

All this holds for any orthogonal base. Then how is a Fourier base special? Really, there isn't all that much special about the Fourier base, except that the coefficients $\alpha_n$ have a neat interpretation: they are the amplitudes of a wave with frequency $f_n$. Choosing to represent a signal by it's Fourier coefficients is just like choosing to ............

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


# Working with the Fourier transform of geometrical objects

Now that we have some understanding of what we're doing, it's time to get out hands dirty. We'll transform an ellipsoid into the frequency domain, play around with the frequencies a bit, and transform it back to see what effect our meddling has had.



```python
import numpy as np

def ellipsoid(theta, phi, r1, r2, r3):
    x = r1 * np.cos(theta) * np.cos(phi)
    y = r2 * np.cos(theta) * np.sin(phi)
    z = r3 * np.sin(theta)
    return x, y, z


r1 = r2 = r3 = 1
signal = np.zeros((360, 360, 3), dtype=np.float)
for theta in np.arange(0, 360):
    for phi in np.arange(0, 360):
    	signal[theta, phi] = ellipsoid(theta, phi, r1, r2, r3)

```


# Conclusion
This was inspiring! In the process of writing this post, I had ideas for several little games to play around with multidimensional Fourier transforms. One that I'll soon put on github goes like this: split a screen in two panes. The left one is a flat surface, on which the user can draw. This surface represents the Fourier amplitudes. With every frame, the drawn amplitudes get transformed back into a tree-dimensional object, which will be displayed on the right pane. Link soon to follow!
