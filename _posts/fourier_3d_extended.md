---
layout: post
title:  "Fourier Transform on Three-Dimensional Objects: Part 2"
date:   2018-03-15 20:09:02 +0100
categories: fourier
---

This post is still in draft stage! I'll finish it up as soon as I find some time to write again.

$$ \newcommand{\Reals}[]{\mathbb{R}} $$
$$ \newcommand{\Complexes}[]{\mathbb{C}} $$
$$ \newcommand{\Naturals}[]{\mathbb{N}} $$
$$ \newcommand{\Geom}[]{\mathbb{G}} $$


Lets consider vector valued functions. There are three ways we can do Fourier transforms on them

 - consider each dimension separately: do a Fourier transform on x, one on y, and one on z
 - make a vectorspace of vectorfields. In the case of ellipsoid, that would be a vectorspace consisting of functions $\Reals^2 \to \Complex^3$. Still have to make up a suitable Fourier base, though.
 - express the ellipsoid in the conformal model of geometric algebra. Thus we deals with a vectorspace of $\Reals^2 \to \Geom$. This is a simple one dimensional transform.
