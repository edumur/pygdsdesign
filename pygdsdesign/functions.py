from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.operation import select_polygon_per_layer
from pygdsdesign.typing_local import Number

import numpy as np
from typing import (
    Tuple,
)


def print_ebeam_time(polygon: PolygonSet,
                     layer: int | dict,
                     beam_current: float,
                     resist_dose: float,
                     datatype: int=0,
                     eos: int=3,
                     stage_displacement_time: float=0.05) -> None:
    """
    Print the exposure time of the layer based on its area, the beam current and
    the resist dose.
    Print the stage displacement time of the layer based on its area, the eos
    mode and the estimated stage displacement time.
    The total time is simply the sum of the exposure and displacement time.

    Args:
        polygon: Polygon from which the layer is extracted
        layer:
            if int: layer we want the ebeam exposure time from
            if dict if a key 'layer': layer we want the ebeam exposure time from
        beam_current: beam current in nA.
        resist_dose: resis dose in uC/cm2
        datatype: layer datatype.
            if argument `layer` is a dict, this argument `datatype` is ignored.
            if argument `layer` is an int, this argument `datatype` is used.
            default, 0.
        eos: EOS mode of the ebeam job.
            Must be either 3 or 6.
            An eos mode 3 imply a stage field of 500x500 um2.
            An eos mode 6 imply a stage field of 62.5x62.5 um2.
        stage_displacement_time: Time taken by the ebeam stage to move from
            field to field in second.
    """
    if isinstance(layer, dict):
        if 'layer' in layer.keys():
            if isinstance(layer['layer'], int):
                l = layer['layer']
            else:
                raise ValueError('The key "layer" of the layer argument must be an int')
        else:
            raise ValueError('Layer argument must be a dict containing a key "layer"')
        if 'datatype' in layer.keys():
            if isinstance(layer['datatype'], int):
                d = layer['datatype']
            else:
                raise ValueError('The key "datatype" of the layer argument must be an int')
        else:
            raise ValueError('Layer argument must be a dict containing a key "datatype"')
    elif isinstance(layer, int):
        l = layer
        d = datatype
    else:
        raise ValueError('Layer argument must be an int')

    # get layer area in um2
    area = select_polygon_per_layer(polygon, layer=l, datatype=d).get_area()

    # exposure time in s
    exposure_time = area/1e8*resist_dose*1e3/beam_current
    minutes = exposure_time//60
    hours = minutes//60

    print('')
    print('++++++++++++++++++++++++++++++++++++')
    print('ebeam info:')
    print('    resist sensitivity: {} uC/cm2'.format(resist_dose))
    print('    area: {:.0f} um2 = {:.1f} mm2'.format(area, area/1e6))
    print('    current: {} nA'.format(beam_current))
    print('    exposure duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, exposure_time % 60))

    # get layer containing area in um2
    dx, dy = polygon.get_size()
    area_c = dx*dy
    # stage displacement time
    if eos==3:
        field_area = 500.*500.
    elif eos==6:
        field_area = 62.5*62.5
    else:
        raise ValueError('eos argument must be either 3 or 6')

    stage_time = area_c/field_area*stage_displacement_time
    minutes = stage_time//60
    hours = minutes//60
    print('    number of field: {:.0f}'.format(area_c/field_area))
    print('    stage displacement duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, stage_time % 60))

    # total time
    total_time = exposure_time + stage_time
    minutes = total_time//60
    hours = minutes//60
    print('    total ebeam duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, total_time % 60))



def distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ) -> float:
    """Return the distance between the two point (x1;y1) and (x2;y2)"""
    return np.sqrt((x1-x2)**2 + (y2-y1)**2)



def hobby(
    points: np.ndarray,
    angles: np.ndarray|None=None,
    curl_start: Number=1,
    curl_end: Number=1,
    t_in: Number=1,
    t_out: Number=1,
    cycle: bool=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate control points for a smooth interpolating curve.

    Uses the Hobby algorithm [1]_ to calculate a smooth interpolating
    curve made of cubic Bezier segments between each pair of points.

    Parameters
    ----------
    points : Numpy array[N, 2]
        Vertices in the interpolating curve.
    angles : array-like[N] or None
        Tangent angles at each point (in *radians*).  Any angles defined
        as None are automatically calculated.
    curl_start : number
        Ratio between the mock curvatures at the first point and at its
        neighbor.  A value of 1 renders the first segment a good
        approximation for a circular arc.  A value of 0 will better
        approximate a straight segment.  It has no effect for closed
        curves or when an angle is defined for the first point.
    curl_end : number
        Ratio between the mock curvatures at the last point and at its
        neighbor.  It has no effect for closed curves or when an angle
        is defined for the last point.
    t_in : number or array-like[N]
        Tension parameter when arriving at each point.  One value per
        point or a single value used for all points.
    t_out : number or array-like[N]
        Tension parameter when leaving each point.  One value per point
        or a single value used for all points.
    cycle : bool
        If True, calculates control points for a closed curve, with
        an additional segment connecting the first and last points.

    Returns
    -------
    out : 2-tuple of Numpy array[M, 2]
        Pair of control points for each segment in the interpolating
        curve.  For a closed curve (`cycle` True), M = N.  For an open
        curve (`cycle` False), M = N - 1.

    References
    ----------
    .. [1] Hobby, J.D.  *Discrete Comput. Geom.* (1986) 1: 123.
       `DOI: 10.1007/BF02187690 <https://doi.org/10.1007/BF02187690>`_
    """
    z = points[:, 0] + 1j * points[:, 1]
    n = z.size
    if np.isscalar(t_in):
        t_in = t_in * np.ones(n)
    else:
        t_in = np.array(t_in)
    if np.isscalar(t_out):
        t_out = t_out * np.ones(n)
    else:
        t_out = np.array(t_out)
    if angles is None:
        angles = [None] * n
    rotate = 0
    if cycle and any(a is not None for a in angles):
        while angles[rotate] is None:
            rotate += 1
        angles = [angles[(rotate + j) % n] for j in range(n + 1)]
        z = np.hstack((np.roll(z, -rotate), z[rotate : rotate + 1]))
        t_in = np.hstack((np.roll(t_in, -rotate), t_in[rotate : rotate + 1]))
        t_out = np.hstack((np.roll(t_out, -rotate), t_out[rotate : rotate + 1]))
        cycle = False
    if cycle:
        # Closed curve
        v = np.roll(z, -1) - z
        d = np.abs(v)
        delta = np.angle(v)
        psi = (delta - np.roll(delta, 1) + np.pi) % (2 * np.pi) - np.pi
        coef = np.zeros(2 * n)
        coef[:n] = -psi
        m = np.zeros((2 * n, 2 * n))
        i = np.arange(n)
        i1 = (i + 1) % n
        i2 = (i + 2) % n
        ni = n + i
        m[i, i] = 1
        m[i, n + (i - 1) % n] = 1
        # A_i
        m[ni, i] = d[i1] * t_in[i2] * t_in[i1] ** 2
        # B_{i+1}
        m[ni, i1] = -d[i] * t_out[i] * t_out[i1] ** 2 * (1 - 3 * t_in[i2])
        # C_{i+1}
        m[ni, ni] = d[i1] * t_in[i2] * t_in[i1] ** 2 * (1 - 3 * t_out[i])
        # D_{i+2}
        m[ni, n + i1] = -d[i] * t_out[i] * t_out[i1] ** 2
        sol = np.linalg.solve(m, coef)
        theta = sol[:n]
        phi = sol[n:]
        w = np.exp(1j * (theta + delta))
        a = 2 ** 0.5
        b = 1.0 / 16
        c = (3 - 5 ** 0.5) / 2
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        alpha = (
            a * (sintheta - b * sinphi) * (sinphi - b * sintheta) * (costheta - cosphi)
        )
        cta = z + w * d * ((2 + alpha) / (1 + (1 - c) * costheta + c * cosphi)) / (
            3 * t_out
        )
        ctb = np.roll(z, -1) - np.roll(w, -1) * d * (
            (2 - alpha) / (1 + (1 - c) * cosphi + c * costheta)
        ) / (3 * np.roll(t_in, -1))
    else:
        # Open curve(s)
        n = z.size - 1
        v = z[1:] - z[:-1]
        d = np.abs(v)
        delta = np.angle(v)
        psi = (delta[1:] - delta[:-1] + np.pi) % (2 * np.pi) - np.pi
        theta = np.empty(n)
        phi = np.empty(n)
        i = 0
        if angles[0] is not None:
            theta[0] = angles[0] - delta[0]
        while i < n:
            j = i + 1
            while j < n + 1 and angles[j] is None:
                j += 1
            if j == n + 1:
                j -= 1
            else:
                phi[j - 1] = delta[j - 1] - angles[j]
                if j < n:
                    theta[j] = angles[j] - delta[j]
            # Solve open curve z_i, ..., z_j
            nn = j - i
            coef = np.zeros(2 * nn)
            coef[1:nn] = -psi[i : j - 1]
            m = np.zeros((2 * nn, 2 * nn))
            if nn > 1:
                ii = np.arange(nn - 1)  # [0 .. nn-2]
                i0 = i + ii  # [i .. j-1]
                i1 = 1 + i0  # [i+1 .. j]
                i2 = 2 + i0  # [i+2 .. j+1]
                ni = nn + ii  # [nn .. 2*nn-2]
                ii1 = 1 + ii  # [1 .. nn-1]
                m[ii1, ii1] = 1
                m[ii1, ni] = 1
                # A_ii
                m[ni, ii] = d[i1] * t_in[i2] * t_in[i1] ** 2
                # B_{ii+1}
                m[ni, ii1] = -d[i0] * t_out[i0] * t_out[i1] ** 2 * (1 - 3 * t_in[i2])
                # C_{ii+1}
                m[ni, ni] = d[i1] * t_in[i2] * t_in[i1] ** 2 * (1 - 3 * t_out[i0])
                # D_{ii+2}
                m[ni, ni + 1] = -d[i0] * t_out[i0] * t_out[i1] ** 2
            if angles[i] is None:
                to3 = t_out[0] ** 3
                cti3 = curl_start * t_in[1] ** 3
                # B_0
                m[0, 0] = to3 * (1 - 3 * t_in[1]) - cti3
                # D_1
                m[0, nn] = to3 - cti3 * (1 - 3 * t_out[0])
            else:
                coef[0] = theta[i]
                m[0, 0] = 1
                m[0, nn] = 0
            if angles[j] is None:
                ti3 = t_in[n] ** 3
                cto3 = curl_end * t_out[n - 1] ** 3
                # A_{nn-1}
                m[2 * nn - 1, nn - 1] = ti3 - cto3 * (1 - 3 * t_in[n])
                # C_nn
                m[2 * nn - 1, 2 * nn - 1] = ti3 * (1 - 3 * t_out[n - 1]) - cto3
            else:
                coef[2 * nn - 1] = phi[j - 1]
                m[2 * nn - 1, nn - 1] = 0
                m[2 * nn - 1, 2 * nn - 1] = 1
            if nn > 1 or angles[i] is None or angles[j] is None:
                # print("range:", i, j)
                # print("A =", m)
                # print("b =", coef)
                sol = np.linalg.solve(m, coef)
                # print("x =", sol)
                theta[i:j] = sol[:nn]
                phi[i:j] = sol[nn:]
            i = j
        w = np.hstack(
            (np.exp(1j * (delta + theta)), np.exp(1j * (delta[-1:] - phi[-1:])))
        )
        a = 2 ** 0.5
        b = 1.0 / 16
        c = (3 - 5 ** 0.5) / 2
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        alpha = (
            a * (sintheta - b * sinphi) * (sinphi - b * sintheta) * (costheta - cosphi)
        )
        cta = z[:-1] + w[:-1] * d * (
            (2 + alpha) / (1 + (1 - c) * costheta + c * cosphi)
        ) / (3 * t_out[:-1])
        ctb = z[1:] - w[1:] * d * (
            (2 - alpha) / (1 + (1 - c) * cosphi + c * costheta)
        ) / (3 * t_in[1:])
        if rotate > 0:
            cta = np.roll(cta, rotate)
            ctb = np.roll(ctb, rotate)
    return (
        np.vstack((cta.real, cta.imag)).transpose(),
        np.vstack((ctb.real, ctb.imag)).transpose(),
    )