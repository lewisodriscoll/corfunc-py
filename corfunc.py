#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fftpack import dct
from scipy.signal import argrelextrema
from numpy.linalg import lstsq
import warnings


class Interpolator(object):
    def __init__(self, f, g, start, stop):
        self.f = f
        self.g = g
        self.start = start
        self.stop = stop
        self._lastx = []
        self._lasty = []

    def __call__(self, x):
        if self._lastx == [] or x.tolist() != self._lastx.tolist():
            self._lasty = self._smoothed_function(x)
            self._lastx = x
        return self._lasty

    def _smoothed_function(self,x):
        ys = np.zeros(x.shape)
        ys[x <= self.start] = self.f(x[x <= self.start])
        ys[x >= self.stop] = self.g(x[x >= self.stop])
        with warnings.catch_warnings():
            # Ignore divide by zero error
            warnings.simplefilter('ignore')
            h = 1/(1+(x-self.stop)**2/(self.start-x)**2)
        mask = np.logical_and(x > self.start, x < self.stop)
        ys[mask] = h[mask]*self.g(x[mask])+(1-h[mask])*self.f(x[mask])
        return ys


# Pretend Python allows for anonymous classes
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def porod(q, K, sigma,bg):
    """Calculate the Porod region of a curve"""
    return bg+(K*q**(-4))*np.exp(-q**2*sigma**2)


def guinier(q, A, B):
    """Calculate the Guinier region of a curve"""
    return A*np.exp(B*q**2)


def fit_guinier(q, iq):
    """Fit the Guinier region of a curve"""
    A = np.vstack([q**2, np.ones(q.shape)]).T
    return lstsq(A, np.log(iq))

def fit_porod(q, iq):
    """Fit the Porod region of a curve"""
    fitp = curve_fit(lambda q, k, sig, bg: porod(q, k, sig, bg)*q**2,
                     q, iq*q**2)[0]
    [k, sigma, bg] = fitp
    return k, sigma, bg


def fit_data(q, iq, lowerq, upperq):
    """
    Given a data set, extrapolate out to large q with Porod
    and to q=0 with Guinier
    """

    mask = np.logical_and(q > upperq[0], q < upperq[1])

    # Returns the values of k, sigma and bg for the best fitting Porod curve
    k, sigma, bg = fit_porod(q[mask], iq[mask])

    # Smooths between the best-fit porod function and the data to produce a
    # better fitting curve
    data = interp1d(q, iq)
    s1 = Interpolator(data, lambda x: porod(x, k, sigma, bg), upperq[0], q[-1])

    mask = np.logical_and(q < lowerq, 0 < q)

    # Returns parameters for the best-fit Guinier function
    g = fit_guinier(q[mask], iq[mask])[0]

    # Smooths between the best-fit Guinier function and the Porod curve
    s2 = Interpolator((lambda x: (np.exp(g[1]+g[0]*x**2))), s1, q[0], lowerq)

    return s2, bg


def corr(f, lowerq, upperq):
    """Transform a scattering curve into a correlation function"""
    orig = np.loadtxt(f, skiprows=1, dtype=np.float32)
    q = orig[:480, 0]
    iq = orig[:480, 1]

    if lowerq <= q.min():
        raise Exception("MINQ must be greater than the lowest q value")
    if upperq[1] > q.max():
        raise Exception("UPQ2 must be less than or equal to the greatest q value")
    if upperq[0] > upperq[1]:
        raise Exception("UPQ1 must be less than UPQ2")

    s2, bg = fit_data(q, iq, lowerq, upperq)
    qs = np.arange(0, q[-1]*100, (q[1]-q[0]))
    iqs = s2(qs)
    transform = dct((iqs-bg)*qs**2)
    transform = transform / transform.max()
    xs = np.pi*np.arange(len(qs),dtype=np.float32)/(q[1]-q[0])/len(qs)

    return (xs, transform)


def extract(x, y):
    """Extract the interesting measurements from a correlation function"""
    # Calculate indexes of maxima and minima
    maxs = argrelextrema(y, np.greater)[0]
    mins = argrelextrema(y, np.less)[0]

    # If there are no maxima, return NaN
    garbage = Struct(minimum=np.nan,
                     maximum=np.nan,
                     dtr=np.nan,
                     Lc=np.nan,
                     d0=np.nan,
                     A=np.nan)
    if len(maxs) == 0:
        return garbage
    GammaMin = y[mins[0]]  # The value at the first minimum

    ddy = (y[:-2]+y[2:]-2*y[1:-1])/(x[2:]-x[:-2])**2  # Second derivative of y
    dy = (y[2:]-y[:-2])/(x[2:]-x[:-2])  # First derivative of y
    # Find where the second derivative goes to zero
    zeros = argrelextrema(np.abs(ddy), np.less)[0]
    # locate the first inflection point
    linear_point = zeros[0]
    linear_point = int(mins[0]/10)

    # Try to calculate slope around linear_point using 80 data points
    lower = linear_point - 40
    upper = linear_point + 40

    # If too few data points to the left, use linear_point*2 data points
    if lower < 0:
        lower = 0
        upper = linear_point * 2
    # If too few to right, use 2*(dy.size - linear_point) data points
    elif upper > dy.size:
        upper = dy.size
        width = dy.size - linear_point
        lower = 2*linear_point - dy.size

    m = np.mean(dy[lower:upper])  # Linear slope
    b = y[1:-1][linear_point]-m*x[1:-1][linear_point]  # Linear intercept

    Lc = (GammaMin-b)/m  # Hard block thickness

    # Find the data points where the graph is linear to within 1%
    mask = np.where(np.abs((y-(m*x+b))/y) < 0.01)[0]
    if len(mask) == 0:  # Return garbage for bad fits
        return garbage
    dtr = x[mask[0]]  # Beginning of Linear Section
    d0 = x[mask[-1]]  # End of Linear Section
    GammaMax = y[mask[-1]]
    A = -GammaMin/GammaMax  # Normalized depth of minimum

    return Struct(minimum=x[mins[0]],
                  maximum=x[maxs[0]],
                  dtr=dtr,
                  Lc=Lc,
                  d0=d0,
                  A=A)

values = []
specs = []


def main(files, lowerq, upperq, export=None, plot=False, save=None):
    """Load a set of intensity curves and gathers the relevant statistics"""
    import os.path

    for f in files:
        x, y = corr(f, lowerq, upperq)
        plt.plot(x, y, label=os.path.basename(f))
        values.append(extract(x, y))
        specs.append(y)
        x0 = x

    plt.xlabel("Distance [Angstroms]")
    plt.ylabel("Correlation")
    plt.axhline(0,color='k')
    plt.xlim([0,200])
    plt.legend()

    if plot:
        plt.show()
    elif save:
        plt.savefig(save)

    from math import isnan

    maxs = np.array([v.maximum for v in values if not isnan(v.minimum)])
    dtrs = np.array([v.dtr for v in values if not isnan(v.minimum)])
    lcs = np.array([v.Lc for v in values if not isnan(v.minimum)])
    qs = np.array([v.d0 for v in values if not isnan(v.minimum)])
    As = np.array([v.A for v in values if not isnan(v.minimum)])

    def printWithError(title, values):
        print(title)
        print("%f ± %f" % (np.median(values),
                            np.max(np.abs(values-np.median(values)))))

    printWithError("Long Period", maxs)
    printWithError("Average Hard Block Thickness", lcs)
    printWithError("Average Interface Thickness", dtrs)
    printWithError("Average Core Thickness ", qs)
    printWithError("PolyDispersity", As)
    printWithError("Filling Fraction", lcs/maxs)

    if export:
        np.savetxt(export,
                   np.vstack([x0, specs]).T)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Perform correlation function analysis on scattering data')
    parser.add_argument('--export', action='store',
                        help='Export the extracted real space data to a file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plot', action='store_true',
                       help='Display a plot of the correlation functions.')
    group.add_argument('--saveImage', action='store',
                       help='Save a plot to an image file.')

    parser.add_argument('FILE', nargs="+",
                        help='Scattering data in two column ascii format')
    parser.add_argument('LOWQ', nargs='+',
                        help='Values less than this will beused for back extrapolation')
    parser.add_argument('UPQ1', nargs='+',
                        help='Lower bound of values to use for forward extrapolation')
    parser.add_argument('UPQ2', nargs='+',
                        help='Upper bound of values to use for forward extrapolation')
    args = parser.parse_args()

    lowerq = float(args.LOWQ[0])
    upperq = (float(args.UPQ1[0]), float(args.UPQ2[0]))

    main(args.FILE, lowerq, upperq, args.export,
         args.plot, args.saveImage)
