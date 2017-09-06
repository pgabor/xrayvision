import numpy as np
import pytest

from ..Visibillity import RHESSIVisibility


class TestRHESSIVisibility(object):

    @pytest.mark.parametrize("N,M,isc,harm,erange,trange,totflux,sigamp,"
                             "chi2,xyoffset,type_string,units,"
                             "atten_state,count",
                             [(65, 65, 1, 3, [5.0, 10.0], [7.0, 19.0],
                               0.7, 0.3, 0.4, [10, 15], "photon", "test",
                               0, 72.5)])
    def test_from_map(self, N, M, isc, harm, erange, trange, totflux, sigamp,
                      chi2, xyoffset, type_string, units, atten_state, count):
        # Calculate uv and vis_in
        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N*M)
        vis_in = np.zeros(N*M, dtype=complex)

        # Creating a RHESSI Visibility
        vis = RHESSIVisibility(uv_in, vis_in, isc, harm, erange, trange,
                               totflux, sigamp, chi2, xyoffset, type_string,
                               units, atten_state, count)

        assert vis.isc == isc
        assert vis.harm == harm
        assert vis.erange == erange
        assert vis.trange == trange
        assert vis.totflux == totflux
        assert vis.sigamp == sigamp
        assert vis.chi2 == chi2
        assert vis.xyoffset == xyoffset
        assert vis.type_string == type_string
        assert vis.units == units
        assert vis.atten_state == atten_state
        assert vis.count == count
