---
layout: post
title:  "Fourier Transform on Three-Dimensional Objects"
date:   2018-02-25 20:09:02 +0100
categories: fourier
---

$$ \newcommand{\Reals}[]{\mathbb{R}} $$

$$ \newcommand{\Naturals}[]{\mathbb{N}} $$

# Motivation: adding an octave to a sphere

Like many people I first got into working with the Fourier transform when learning to work with music. It was neat how you could take a signal, find out what notes it consisted of and then adjust the signal to your liking by doing simple operations in the Fourier domain. For example you could add an octave on top of the most important note by just adding another frequency to the signal.

Really, a musical signal is only just a function of time, or, in more technical terms, a mapping \\(\Reals \to \Reals \\). However, there is no reason why we should limit ourselves to only such functions. What about mappings of the form \\( \Reals^n \to \Reals \\), or \\( \Reals^n \to \Reals^m \\)?

This opens up a whole new world of interesting transformations. For example, grayscale pictures can be viewed as a function mapping x- and y-coordinates onto a pixel-value between 0 and 255. This is one example of a mapping of the form \\( \Naturals^2 \to [0, 255] \\).

Geometrical objects make for another interesting example. We can transform any geometrical object that has a parameterized equation. Consider for example the spere:

$$ x = r \cos \theta \sin \phi $$

$$ y = r \sin \theta \sin \phi $$

$$ z = r \cos \theta $$

That means we can express a sphere as a vector-valued function of the form \\( \Reals \times [0, 2 \pi]^2 \to \Reals^3 \\).

What I initially found fascinating about the Fourier transform was how you could modify musical signals. Now that we have learned that you can just as well apply Fourier to geometrical objects, let's find out what it means to modify them in the Fourier domain! Is there such a thing as "adding an octave to a spere"? If so, what would the spere look like?


# A little bit of theory

Let's first consider the one-dimensional case. There is an infinite amount of Fourier base-functions of the form 

$$ f_n(x) = ... $$

These base-functions form a base for the vector-space of sqare-integrable, periodic functions \\( \Reals \to \Reals \\). 
The sphere, however, is one function in the space of functions mapping \\( \Reals^3 \to \Reals^3 \\). This space, too, has a Fourier base, and that base consists of functions of the form 

$$ f_n(x) = ... $$

# Transforming and modifying a sphere
