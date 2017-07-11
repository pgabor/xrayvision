"""Submodule to provide mock visibility data for XRayVision
"""

__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "Licensed under a 3-clause BSD style license - see LICENSE.rst"

import numpy as np

def get_uniform_complex_data(xdim : int = 1, ydim :int = 1, value : complex = 0+0j) -> np.array:
    """
    Creates mock data for tests. The result is a numpy array, which has the specified size
    and every entry contains the given value
    
    Parameters
    ----------
    xdim : int
        x size of the array
    ydim : int
        y size of the array
    value : complex
        the result will contain this value
    
    Returns
    -------
    data : np.array
        A numpy array with the specified shape and value
    """
    if (xdim <=0 or ydim <= 0):
        raise ValueError("xdim and ydim have to be greater than zero!")
    data = np.repeat(value, xdim * ydim)
    data = data.reshape(xdim, ydim)
    return data
