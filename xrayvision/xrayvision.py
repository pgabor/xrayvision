"""Module for synthesis imaging
"""

__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "Licensed under a 3-clause BSD style license - see LICENSE.rst"

import numpy as np


class XRayVision:
    """
    XRayVision(observed: str, )
    
    Helper class for synthesis imaging
    
    Parameters
    ----------
    observed : str
        The observed quantity. For example energy of photons or electrons, count of incidents, etc... (the default is "Not defined")             
    xyOffset : np.array([x, y])
        Offset from the centre. Should be provided as a 2D numpy array (the default is [0.0, 0.0])
    detectorCounts : float
    
    attenuatorState : bool
        State of the attenuator (the default is True)
    harmonic : bool
        True if harmonics used, otherwise False (the default is True)
    isc: bool
    
    Attributes
    ----------
    timeRange : np.array
        Numpy array in the following format: [t_min, t_max]
    energyRange : np.array
        Numpy array in the following format: [E_min, E_max]
    totalFlux: float
        The total flux of the data
    error: float
        Error on the visibility
    
    Examples
    ----------
    
    
    See Also
    ----------
    
    
    References
    ----------
    """
    def __init__(self, observed: str  = "Not defined", xyOffset: np.array = np.array([0., 0.]), detectorCounts: float  = 0.,
                 attenuatorState: bool = True, harmonic: bool = False, isc: bool = False):
        if xyOffset.size != 2:
            raise ValueError("xyOffset should contain two entries!")
        
        self.observed = observed
        self.xyOffset = np.array(xyOffset)
        self.detectorCounts = detectorCounts
        self.attenuatorState = attenuatorState
        self.harmonic = harmonic
        self.isc = isc
        
        self.timeRange = np.array([0., 0.])
        self.energyRange = np.array([0., 0.])
        self.totalFlux = 0.
        self.error = 0.
