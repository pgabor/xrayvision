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

class visibility:
    def __init__(self):
        self.visibility = 0.0j
        self.k = (0, 0)

def inner_fourier(data: np.array, k: np.array, sign: float):
    """
    This function returns the inner part of the 2D DFT for dft

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
        dft()

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


def inner_fourier_for_bag(data: np.array, k: np.array, dim: tuple):
    """
    This function returns the inner part of the 2D DFT for dft
    from visibility bag

    Parameters
    ----------
    data : np.array
        List of the known visibilities

    k : np.array
        The result shuld be calculated for this

    dim : tuple
        The dimensions of the desired image

    Returns
    -------
        The corresponding value for k as a result of Fourier transformation

    See Also
    --------
        dft()

    Notes
    -----

    Reference
    ----------
    """
    result = 0.0j
    for i in data:
        n = np.array([complex(i.k[0]), complex(i.k[1])])
        q = np.divide(k, np.array([complex(dim[0]), complex(dim[1])]))
        inside = np.dot(np.multiply(n, 2.0j), np.transpose(q))
        result += np.exp(inside) * i.visibility
    return result


def dft(data, sign):
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


def dft_from_bag(vis, dim):
    """
    This function returns with the 2D DFT for a given visibility bag

    Parameters
    ----------
    vis : np.array
        List of the visibilities

    dim : tuple
        Contains the desired size of the final image

    Returns
    -------
        Complex numpy array with the calculated Fourier-transformed data

    See Also
    --------
        dft()

    Notes
    -----

    Reference
    ----------
    """
    result = np.zeros(dim, dtype=np.complex)
    for i in range(dim[0]):
        for j in range(dim[1]):
            result[i, j] = inner_fourier_for_bag(vis, np.array([i, j]), dim)
    return result

def create_visibility_bag(positions: np.array, data: np.array):
    """
    Creates a visibility bag from the given positions

    Parameters
    ----------
    positions : np.array
        Calculate the visibilities there
    data : np.array
        The intensity data

    Returns
    -------
        np.array with visibility objects for the given positions

    See Also
    --------

    Notes
    -----

    Reference
    ----------
    """
    result = []
    for i in positions:
        res = visibility()
        res.visibility = inner_fourier(data, np.array([i]), -1.0)
        res.k = i
        result.append(res)
    return result

def example1():
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
        dft()

    Notes
    -----

    Reference
    ----------
    """
    im = np.ones((15, 15))
    im[10, 5] = 100
    im[10, 10] = 150
    vis = dft(im, -1.0)
    plt.imshow(im)
    pylab.show()
    plt.imshow(np.real(vis))
    pylab.show()
    plt.imshow(np.imag(vis))
    pylab.show()
    im_b = dft(vis, 1.0)
    plt.imshow(np.real(im_b))
    pylab.show()


def example2():
    """
    This function creates an "image" and tries to convert it twice,
    it shows a plot for each intermediate step - visibility bag

    Parameters
    ----------

    Returns
    -------
        Nothing

    See Also
    --------
        dft_from_bag()

    Notes
    -----

    Reference
    ----------
    """
    im = np.ones((15, 15))
    im[10, 5] = 100
    im[10, 10] = 150
    vis = create_visibility_bag(np.array([[1, 1], [1, 13], [13, 1], [13, 13]]), im)
    plt.imshow(im)
    im_b = dft_from_bag(vis, (15, 15))
    plt.imshow(np.real(im_b))
    pylab.show()
