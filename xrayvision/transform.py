"""DFT for image and visibilities

"""
#
# TODO
#
__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "xxx"

import numpy as np
import matplotlib.pyplot as plt
import pylab


def inner_fourier(data: np.array, k: np.array, sign: float):
    """
    This function returns the inner part of the 2D DFT for outer_fourier

    Parameters
    ----------
    data : np.array
        np.array of the data, they can be visibilities or intensities

    k : np.array
        The result shuld be calculated for this

    sign : float
        -1.0 for converting from the image space to Fourier-space, 1.0 in
        the other direction 

    Returns
    -------
        The corresponding value for k as a result of Fourier transformation

    See Also
    --------
        outer_fourier()

    Notes
    -----

    Reference
    ----------
    """
    result = 0.0j
    dim = data.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            n = np.array([complex(i), complex(j)])
            q = np.divide(k, np.array([complex(dim[0]), complex(dim[1])]))
            inside = np.dot(np.multiply(n, 2.0j * sign), np.transpose(q))
            result += np.exp(inside) * data[i, j]
    return result


def outer_fourier(data, sign):
    """
    This function returns with the 2D DFT for the given data with the
    appropiate conversation direction

    Parameters
    ----------
    data : np.array
        np.array of the data, they can be visibilities or intensities

    sign : float
        -1.0 for converting from the image space to Fourier-space, 1.0 in
        the other direction 

    Returns
    -------
        Complex numpy array with the calculated Fourier-transformed data

    See Also
    --------
        inner_fourier()

    Notes
    -----

    Reference
    ----------
    """
    result = np.zeros(data.shape, dtype=np.complex)
    dim = data.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i, j] = inner_fourier(data, np.array([i, j]), sign)
    return result


def example():
    """
    This function creates an "image" and tries to convert it twice,
    it shows a plot for each intermediate step

    Parameters
    ----------

    Returns
    -------
        Nothing

    See Also
    --------
        outer_fourier()

    Notes
    -----

    Reference
    ----------
    """
    im = np.ones((15, 15))
    im[10, 5] = 100
    im[10, 10] = 150
    vis = outer_fourier(im, -1.0)
    plt.imshow(im)
    pylab.show()
    plt.imshow(np.real(vis))
    pylab.show()
    plt.imshow(np.imag(vis))
    pylab.show()
    im_b = outer_fourier(vis, 1.0)
    plt.imshow(np.real(im_b))
    pylab.show()
