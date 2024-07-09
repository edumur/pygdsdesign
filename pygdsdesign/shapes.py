import numpy as np
import copy
import os
from typing import Optional, Dict, Any
import warnings

from pygdsdesign.library import GdsLibrary
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.polygons import Rectangle, RectangleCentered, Text
from pygdsdesign.operation import boolean, offset, merge


def crosses(coordinates: list,
            layer: int = 1,
            datatype: int = 0,
            width: float=5,
            h_length: float=35,
            v_length: float=35) -> PolygonSet:
    """
    Returns a polygon containing crosses at all coordinates specified in the
    coordinates list.

    Args:
        coordinates (list): List of tupples with coordinats.
        layer (int, optional): gds layer of the crosses. Defaults to 1.
        datatype (int, optional): gds datatype of the crosses. Defaults to 1.
        width (float, optional): Width of the arms of a single cross.
        Defaults to 5.
        h_length (float, optional): Total horizantal length of single cross,
        includes the width of the arm. Defaults to 35.
        v_length (float, optional): Total vertical length of single cross,
        includes the width of the arm. Defaults to 35.

    Returns:
        PolygonSet: Set of polygons containing crosses at positions
        specified by coordinates.
    """
    crosses = PolygonSet(layers=[layer], datatypes=[datatype])
    for coord in coordinates:
        cr = cross(layer=layer, datatype=datatype, width=width, h_length=h_length, v_length=v_length)
        cr.translate(coord[0], coord[1])
        crosses+=cr
    return crosses


def cross(layer: int=1,
          datatype: int=1,
          width: float=5,
          h_length: float=35,
          v_length: float=35) -> PolygonSet:
    """
    Returns a cross, specified by width, h_length and v_length.

    Args:
        layer (int, optional): gds layer of the cross. Defaults to 1.
        datatype (int, optional): gds datatype of the cross. Defaults to 1.
        width (float, optional): Width of the arms of the cross. Defaults to 5.
        h_length (float, optional): Total horizontal length of the cross,
        includes the width of the arm. Defaults to 35.
        v_length (float, optional): Total vertical length of the cross,
        includes the width of the arm. Defaults to 35.

    Returns:
        Polygon: Polygon containing the cross.
    """
    cross = PolygonSet(layers=[layer], datatypes=[datatype])
    cross += Rectangle((-h_length/2, -width/2), (h_length/2, width/2),
                       layer=layer, datatype=datatype)
    cross += Rectangle((-width/2, -v_length/2), (width/2, v_length/2),
                       layer=layer, datatype=datatype)
    merge(cross.center())
    return cross


def global_marks_ebeam(w: float=10,
                       l: float=200,
                       directional_structures: bool=True,
                       directional_offset: float=40,
                       directional_structures_length: float=5,
                       directional_scaling: float=4,
                       squared_center: bool=False,
                       layer: int=1,
                       datatype: int=1,
                       color: str='',
                       name: str='') -> PolygonSet:
    """
    Function that returns an ebeam alignment mark for global alignment.
    When directional_structures is True, triangular structures pointing to the
    center of the mark are added to help finding the center of the cross with
    the ebeam system.

    Args:
        w: cross width in um.
            Defaults to 10.
        l: cross length in um.
            Defaults to 200.
        directional_structures: when True, structures are added to show the center
            of the cross, helping to find the cross on the ebeam system.
            Defaults to True.
        directional_offset: minimal distance between the directional mark and the
            cross. Should be large enough to avoid interferences while doing the
            mark detection
            Defaults to 40.
        directional_structures_length: Length of the directional mark in um.
            The mark beeing a triangle, this length corresponds to its base.
            Defaults to 5.
        directional_scaling: Scaling factor applied to the structure.
            Smaller size are closer to the center.
            Defaults to 4.
        squared_center: If True, add two hollow squares in diagonal at the center
            of the cross.
            It offers a easy target to pinpoint while doing manual alignement.
            Example below when squared cented is true:
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
                          %%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%
            Defaults to False.
        layer: gds layer of chip marks.
            Defaults to 1.
        datatype: gds datatype of chip marks.
            Defaults to 1.
        color: gds color of chip marks.
            Defaults to ''.
        name: gds name of chip marks.
            Defaults to ''.
    """

    # Make the crosse
    cross = PolygonSet()
    cross += Rectangle((0, 0), (w, l), layer=layer, datatype=datatype, color=color, name=name).center()
    cross += Rectangle((0, 0), (l, w), layer=layer, datatype=datatype, color=color, name=name).center()

    # By default the total structure is just the cross
    tot = copy.deepcopy(cross)

    if squared_center:
        temp = Rectangle((0, 0), (w/2, w/2))
        temp += Rectangle((0, 0), (-w/2, -w/2))
        tot = boolean(cross,
                      temp,
                      'xor', layer=layer, datatype=datatype, color=color, name=name)

        # If the boolean failed, we return the cross without error and inform the user
        if tot is None:
            tot = cross
            warnings.warn('You asked for crossed center ebeam marked but it failed. Please check your input dimensions.')


    if directional_structures:
        temp = PolygonSet()

        # Create a default triangle with the proper orientation
        t = PolygonSet([[(0, 0), (directional_structures_length, 0), (directional_structures_length/2, 2*directional_structures_length)]])
        t.rotate(np.pi/2, t.center())

        # Add many triangles with the proper rotation in 10 concentrics circles
        for r, s in zip(np.linspace(directional_offset, l*0.75, 10),
                        np.linspace(1, directional_scaling, 10)):
            p = 2*np.pi*r
            nb_p = int(p/30)
            for theta in np.linspace(0, 2*np.pi, nb_p):
                z = r*np.exp(1j*theta)
                temp += copy.copy(t).scale(s).rotate(theta).translate(z.real, z.imag)

        # We remove the triangles being too close to the cross
        temp2 = boolean(temp,
                        offset(tot, directional_offset),
                        'not')

        # We remove the triangles being outside of the bounding_box of the cross
        temp3 = boolean(temp2,
                        Rectangle((-l/2, -l/2), (l/2, l/2)),
                        'and',
                        layer=layer, datatype=datatype, color=color, name=name)

        # In case the boolean operation return nothing (too small cross for instance)
        if temp3 is None:
            return PolygonSet()

        # We remove the triangles which have been cut from previous boolean
        # operation and are now too small
        temp4 = PolygonSet()
        for p in temp3.polygons:
            t=PolygonSet([p], layers=[layer], datatypes=[datatype], colors=[color], names=[name])
            if t.area()>0.9*directional_structures_length*directional_structures_length:
                temp4 += t
        tot += temp4

    return tot


def chip_marks_ebeam(layer: int=1,
                     datatype: int=1) -> PolygonSet:
    """
    Returns a set of ebeam chip marks.

    Args:
        layer (int, optional): gds layer of chip marks. Defaults to 1.
        datatype (int, optional): gds datatype of chip marks. Defaults to 1.

    Returns:
        PolygonSet: Set of polygons containing ebeam chip marks.
    """
    cross = Rectangle((-7.5, -0.5), (7.5, 0.5),
                            layer=layer, datatype=datatype)
    cross += Rectangle((-0.5, -7.5), (0.5, 7.5),
                             layer=layer, datatype=datatype)
    crosses = PolygonSet()
    crosses += copy.copy(cross).translate(-20, -20)
    crosses += copy.copy(cross).translate(-20, 20)
    crosses += copy.copy(cross).translate(20, 20)
    crosses += copy.copy(cross).translate(20, -20)
    all_crosses = copy.copy(crosses)
    all_crosses += copy.copy(crosses).translate(132.5, 127.5)
    all_crosses += copy.copy(crosses).translate(-132.5, 127.5)
    all_crosses += copy.copy(crosses).translate(132.5, -127.5)
    all_crosses += copy.copy(crosses).translate(-132.5, -127.5)
    return all_crosses


def chip_marks_laser(layer: int=1,
                     datatype: int=1,
                     color: str='',
                     name: str='',
                     only_square: bool=False) -> PolygonSet:
    """
    Returns a set of alignement marks use by some people in optical lithography.
    Consists of a cross with a small square:

                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@
                              @@@@@@@@@@                  @@@@@@@@@
                              @@@@@@@@@@                  @@@@@@@@@
                              @@@@@@@@@@                  @@@@@@@@@
                              @@@@@@@@@@                  @@@@@@@@@
                                                 @@@@@@@@@
                                                 @@@@@@@@@
                                                 @@@@@@@@@
                                                 @@@@@@@@@


    Args:
        layer: gds layer of chip marks. Defaults to 1.
        datatype: gds datatype of chip marks. Defaults to 1.
        color: gds color of chip marks. Defaults to ''.
        name: gds name of chip marks. Defaults to ''.
        only_square: If True, return only the square of the alignement mark.
            Defaults to False.

    Returns:
        PolygonSet: Set of polygons containing laser chip marks.
    """

    # cross
    # The cross is return only if only_square is False (default)
    if only_square:
        p = PolygonSet([[(0, 0)]], [layer], [datatype], [name], [color])
    else:
        p = PolygonSet([[( -5, -10),
                         ( -5,   0),
                         (-15,   0),
                         (-15,   5),
                         ( -5,   5),
                         ( -5,  15),
                         (  0,  15),
                         (  0,   5),
                         (10,    5),
                         (10,    0),
                         (  0,   0),
                         (  0,-10)]], [layer], [datatype], [name], [color])

    ## 2 small squares
    # bottom left
    s1 = PolygonSet([[( 5, -15),
                  ( 5, -10),
                  (10, -10),
                  (10, -15)]], [layer], [datatype], [name], [color])

    # top right
    s2 = PolygonSet([[(10, -10),
                  (10,  -5),
                  (15,  -5),
                  (15, -10)]], [layer], [datatype], [name], [color])

    return p+s1+s2


def crossover(layer_dielectric: int=1,
              dt_dielectric_undercut: int=1,
              layer_metal: int=2,
              name_metal: str='',
              name_dielectric: str='',
              m: float=1,
              w: float=6,
              l: float=120,
              u: float=0) -> PolygonSet:
    """
    Returns a single crossover. Size of the dielectric is in all directions
    bigger by "m" um compared to metal bridge.

    Parameters
    ----------
    layer_dielectric : int, optional
        gds layer of the dielectirc, by default 1
    layer_metal : int, optional
        gds layer of the metal bridge, by default 2
    name_metal : str, optional
        gds layer name of the metal bridge, by default ''
    name_dielectric : str, optional
        gds layer name of the dielectirc, by default ''
    dt_dielectric_undercut : int, optional
        gds datatype of the undercut for the dielectric, by default 1
    m : float, optional
        margin between dielectric and metal in units of um, by default 1
    w : float, optional
        width of the metal crossover in units of um, by default 6
    l : float, optional
        length of the metal crossover in units of um, by default 120
    u : float, optional
        undercut of dielectric in units of um, by default 1

    Returns:
        PolygonSet: Set of polygons containing the crossover.
    """
    # define the metal part
    met = PolygonSet()
    met += Rectangle((0, 0), (4*m + w, 8*m + 3*w), layer=layer_metal, name=name_metal)
    met += Rectangle((4*m + w, 4*m + w), (4*m + w + l, 4*m + 2*w), layer=layer_metal, name=name_metal)
    met += Rectangle((4*m + w + l, 0), (8*m + 2*w + l, 8*m + 3*w), layer=layer_metal, name=name_metal)
    # define the dielectric part
    dielec = PolygonSet()
    dielec += Rectangle((m, m), (3*m + w, 7*m + 3*w), layer=layer_dielectric, name=name_dielectric)
    dielec += Rectangle((3*m + w, 3*m + w), (5*m + w + l, 5*m + 2*w), layer=layer_dielectric, name=name_dielectric)
    dielec += Rectangle((5*m + w + l, m), (7*m + 2*w + l, 7*m + 3*w), layer=layer_dielectric, name=name_dielectric)
    # define the mask to generate the undercut
    undercut = PolygonSet()
    undercut += Rectangle((m - u, m - u), (3*m + w + u, 7*m + 3*w + u), layer=layer_dielectric, name=name_dielectric, datatype=dt_dielectric_undercut)
    undercut += Rectangle((3*m + w - u, 3*m + w - u), (5*m + w + l + u, 5*m + 2*w + u), layer=layer_dielectric, name=name_dielectric, datatype=dt_dielectric_undercut)
    undercut += Rectangle((5*m + w + l - u, m - u), (7*m + 2*w + l + u, 7*m + 3*w + u), layer=layer_dielectric, name=name_dielectric, datatype=dt_dielectric_undercut)
    # get undercut
    temp = boolean(undercut, dielec,
                   'not',
                   layer=layer_dielectric,
                   datatype=dt_dielectric_undercut,
                   name=name_dielectric)

    tot = PolygonSet()
    tot+=met
    tot+=dielec
    if temp is not None: # in case undercut is u=0
        tot+=temp
    return merge(tot).center()


def daisychain(num: int=5,
               layer_dielectric: int=1,
               dt_dielectric_undercut: int=1,
               layer_metal: int=2,
               layer_NbN_etch: int=3,
               name_metal: str='',
               name_dielectric: str='',
               name_NbN_etch: str='',
               m: float=1,
               w: float=4,
               l: float=30,
               d: float=3,
               gap: float=50,
               b: float=200,
               u: float=0) -> PolygonSet:
    """
    Returns a daisychain of "num" crossover in a linear chain with bond pads at each end.

    Parameters
    ----------
    num : int, optional
        Number of crossovers in a linear chain, by default 5
    layer_NbN_etch : int, optional
        gds layer of NbN etching, by default layer_NbN_etch
    layer_metal : int, optional
        gds layer of the metal bridge , by default layer_metal
    layer_dielectric : int, optional
        gds layer of the dielectric, by default layer_dielectric
    name_metal : str, optional
        gds layer name of the metal bridge, by default ''
    name_dielectric : str, optional
        gds layer name of the dielectirc, by default ''
    name_NbN_etch : str, optional
        gds layer name of the  NbN etching, by default ''
    dt_dielectric_undercut : int, optional
        gds datatype of the undercut for the dielectric, by default 1
    m : int, optional
        margin betwewn dielectric and metal in untis of um, by default 1
    w : int, optional
        width of the metal crossover in units of um, by default 4
    l : int, optional
        length of the metal crossover in units of um, by default 30
    d : float, optional
        distance between two crossovers in untis of um, by default 3
    gap : float, optional
        gap etched into NbN in untis of um, by default 50
    b : float, optional
        size of square defining the bond pad in untis of um, by default 200
    u : float, optional
        undercut of dielectric in units of um, by default 1

    Returns:
        PolygonSet: Set of polygons containing the daisychain.
    """
    # create chain of crossovers
    chain = PolygonSet()
    for i in range(num):
        unit = crossover(m=m, l=l, w=w, u=u, layer_dielectric=layer_dielectric, dt_dielectric_undercut=dt_dielectric_undercut, layer_metal=layer_metal, name_dielectric=name_dielectric, name_metal=name_metal)
        unit += Rectangle((5*m + w, 0), (3*m + w + l, 10*m + 3 * w), layer=layer_NbN_etch, name=name_NbN_etch).center()
        dx, dy = unit.get_size()
        chain += unit.translate(i * (dx + d), 0)
    ll, tr = chain.get_bounding_box()
    chain += Rectangle((ll), (tr[0], ll[1] - gap), layer=layer_NbN_etch, name=name_NbN_etch)
    chain += Rectangle((ll[0], tr[1] + gap), (tr), layer=layer_NbN_etch, name=name_NbN_etch)
    chain.center()

    # get bounding box again to add bonding pads
    ll, tr = chain.get_bounding_box()
    dx, dy = chain.get_size()
    # construct first pad
    pad = Rectangle((-gap,-gap - b/2), (0, gap + b/2), layer=layer_NbN_etch, name=name_NbN_etch)
    pad += Rectangle((-gap,-gap - b/2), (b + gap, - b/2), layer=layer_NbN_etch, name=name_NbN_etch)
    pad += Rectangle((-gap, gap + b/2), (b + gap, + b/2), layer=layer_NbN_etch, name=name_NbN_etch)
    pad += Rectangle((b + gap, gap + b/2), (b, dy/2 - gap), layer=layer_NbN_etch, name=name_NbN_etch)
    pad += Rectangle((b + gap, -gap - b/2), (b, -dy/2 + gap), layer=layer_NbN_etch, name=name_NbN_etch)

    # construct second pad and move both in correct position
    pad2 = copy.copy(pad).rotate(np.pi)
    pad2.translate(+dx/2 + b, 0)
    pad.translate(-dx/2 - b, 0)
    chain += pad
    chain += pad2
    t = Text('l = {} um, w = {} um, {}x'.format(l, w, num), 30, layer=layer_metal, name=name_metal)
    t.center().translate(0, 200)
    chain += t

    return merge(chain)


def lateqs_logo(layer: int=0,
                datatype: int=0,
                name: str='',
                color: str='',
                width: float=1000,
                center: bool=False) -> PolygonSet:
    """
    Return the LaTEQS logo in a PolygonSet.

    Args:
        layer (int, optional): layer of the return polygons. Defaults to 0.
        datatype (int, optional): datatype of the return polygons. Defaults to 0.
        width (float, optional): total width of the return polygons. Defaults to 1000.
        center (bool, optional): If the logo is centered to (0, 0).
            Defaults to False, meaning bottom-left is (0,0).

    Returns:
        (PolygonSet): LaTEQS logo in a PolygonSet.
    """

    lib = GdsLibrary()

    path = os.path.join(os.path.dirname(__file__), 'gds', 'lateqs_logo.gds')
    lib.read_gds(infile=path)

    main_cell = lib.top_level()[0]
    tot = PolygonSet()
    for p in main_cell.get_polygonsets():

        tot += PolygonSet(p.polygons,
                          layers=[layer],
                          datatypes=[datatype],
                          names=[name],
                          colors=[color])

    # Rescale logo to given width (height calculated automatically)
    w, h = tot.get_size()
    tot.scale(width/w)

    if center:
        tot.center()

    return tot


def qubit_layer_42(layer: int=42,
                   datatype: int=0) -> PolygonSet:
    """
    Generates layer 42 of the Qubit mask (post-pro alignment marks).
    The global marks are not exactley the same as on the original mask, but
    their center is the same.

    Parameters
    ----------
    layer : int, optional
        GDS layer, by default 42
    datatype : int, optional
        GDS datatype, by default 0

    Returns
    -------
    PolygonSet
        Set of polygons containing layer 42 of the qubit mask.
    """

    layer42 = PolygonSet(layers=[layer], datatypes=[datatype])

    # list of coordinates of global marks
    LETI_global_mark_coordinates = [(-5655, 6990), (65, 6990), (5655, 6990),
                                    (-5655, 4990), (65, 4990), (5655, 4990),
                                    (-5655, 950), (65, 950), (5655, 950),
                                    (-5655, -1050), (65, -1050), (5655, -1050),
                                    (-5655, -5090), (65, -5090), (5655, -5090),
                                    (-5655, -7090), (65, -7090), (5655, -7090),]

    # list of coordinates of the center of the first chip marks in scribe
    LETI_chip_mark_coordinates = [(-1363.5, 7306.5), (1496.5, 7306.5),
                                  (-2663.5, 4286.5), (66.5, 4286.5), (2666.5, 4286.5),
                                  (-1363.5, 1266.5), (1496.5, 1266.5),
                                  (-2663.5, -1753.5), (66.5, -1753.5), (2666.5, -1753.5),
                                  (-1363.5, -4773.5), (1496.5, -4773.5)]

    # generate the global marks
    LETI_global_marks = crosses(coordinates=LETI_global_mark_coordinates,
                                layer=layer,
                                datatype=datatype,
                                width=10,
                                h_length=380,
                                v_length=1000)
    layer42 += LETI_global_marks

    # generate the chip marks (crosses and squares)
    for coord in LETI_chip_mark_coordinates:
        # generate the crosses
        LETI_chip_marks=PolygonSet(layers=[layer], datatypes=[datatype])
        LETI_chip_mark = cross(layer=layer,
                               datatype=datatype,
                               width=1,
                               h_length=16,
                               v_length=16).center()
        LETI_chip_mark += copy.copy(LETI_chip_mark).translate(40,0)
        LETI_chip_mark.center()
        LETI_chip_mark += copy.copy(LETI_chip_mark).translate(0,40)
        LETI_chip_mark.center()
        for i in range(11):
            LETI_chip_marks += copy.copy(LETI_chip_mark).translate(0, -i * 260)
        layer42 += LETI_chip_marks.translate(coord[0], coord[1])

        # generate the squares
        square = Rectangle((0,0), (8,8), layer=layer, datatype=datatype)
        squares=PolygonSet(layers=[layer], datatypes=[datatype])
        for i in range(4):
            for j in range(4):
                squares += copy.copy(square).translate(i * 16, j * 16)
        squares.center()
        squares.translate(0, -130).translate(-1.5, -1.5)
        squares_final = PolygonSet(layers=[layer], datatypes=[datatype])
        for i in range(11):
            squares_final += copy.copy(squares).translate(0, -i * 260)
        layer42 += squares_final.translate(coord[0], coord[1])

    return layer42


def resistivity_4_probes(layer: int=0,
                         name: str='',
                         datatype: int=0,
                         color: str='',
                         pad_width: float=400,
                         current_length: float=4000,
                         current_width: float=80,
                         voltage_length: float=400,
                         voltage_width: float=40,
                         gap:Optional[float]=None,
                         centered: bool=True) -> PolygonSet:
    """
    Return a structure dedicated to a 4 probes DC measurement

    Args:
        layer: Number of the metal layer.
            Defaults to 0.
        name: Name of the metal layer.
            Defaults to ''.
        color: Color of the metal layer.
            Defaults to ''.
        datatype: Datatype of the metal layer.
            Defaults to 0.
        pad_width: Width of the bonding pad. The bonding bad is a square.
            In um.
            Defaults to 400.
        current_length: Effective length of the current line.
            In um.
            This is the length measured by the voltage probe, not the total line length.
            The total line length will be current_length + 2*voltage_length
            Defaults to 4000.
        current_width: Width of the current line.
            Must be much smaller than the current length.
            In um.
            Defaults to 80.
        voltage_length: Length of the voltage probes line.
            In um.
            Defaults to 400.
        voltage_width: Width of the voltage probes line.
            Must be much smaller than the current length.
            In um.
            Defaults to 40.
        gap: If not None, return the surrounding gap through offset and boolean
            operation.
            In um.
            Defaults to None.
        centered: If True, centered the structure.
            Defaults to True.
    """

    _layer: Dict[str, Any] = {'layer'    : layer,
                              'datatype' : datatype,
                              'name'     : name,
                              'color'    : color}

    tot = PolygonSet(polygons=[[(0,0)]], **_layer)

    # Make the current line
    tot += Rectangle((0, 0), (pad_width, pad_width), **_layer)
    tot += Rectangle((0, 0), (current_length+2*voltage_length, current_width), **_layer).translate(pad_width, pad_width/2-current_width/2)
    tot += Rectangle((0, 0), (pad_width, pad_width), **_layer).translate(pad_width+current_length+2*voltage_length, 0)

    # Add the 1st voltage probe
    tot += Rectangle((0, 0), (pad_width, pad_width), **_layer).translate(pad_width+voltage_length-pad_width/2, pad_width/2+current_width/2+voltage_length)
    tot += Rectangle((0, 0), (voltage_width, voltage_length), **_layer).translate(pad_width+voltage_length-voltage_width/2, pad_width/2+current_width/2)

    # Add the 2nd voltage probe
    tot += Rectangle((0, 0), (pad_width, pad_width), **_layer).translate(pad_width+voltage_length+current_length-pad_width/2, pad_width/2+current_width/2+voltage_length)
    tot += Rectangle((0, 0), (voltage_width, voltage_length), **_layer).translate(pad_width+voltage_length+current_length-voltage_width/2, pad_width/2+current_width/2)

    # Get the surrounding gap through offset and boolean operation
    if gap is not None:
        temp = boolean(tot,
                       offset(tot,
                              gap,
                              join_first=True,
                              **_layer),
                       'xor',
                       **_layer)

        if temp is not None:
            tot = PolygonSet(polygons=temp.polygons,
                             layers=temp.layers,
                             datatypes=temp.datatypes,
                             names=temp.names,
                             colors=temp.colors)

    if centered:
        return tot.center()
    else:
        return tot


def qubit_layer_19(
    layer: int = 19, layer_annotation: int = 49, datatype: int = 0
) -> PolygonSet:
    """
    Generates part of the layer 19 (via active to Metal 1) of the Qubit mask LETI.
    The device vias generates are:
    - 4G11_1
    - 4G11_2
    - 4G11_3
    - 4G21_1
    - 4G21_2
    - 4G22_1
    - 4G22_2
    - 4G23_1
    - 4G23_2
    - 8G11
    - 7G11_1
    - 7G11_2

    Each device is associated with a Text (0.5 micron above the top left via) with the device name in layer_annoation (layer 49 by default)

    Parameters
    ----------
    layer : int, optional
        GDS layer, by default 19
    layer_annotation: int, optinal
        GDS layer, by default 49
    datatype : int, optional
        GDS datatype, by default 0

    Returns
    -------
    PolygonSet
        Set of polygons containing part of the layer 19 of the LETI qubit mask
        + text annotation to locate devices in layer_annotation (default 49)
    """

    # dictonary with the via coordinate (center) of different devices

    devices = {
        "4G11_1": [
            (-1835.305, 4023.675),
            (-1834.96, 4023.695),
            (-1834.615, 4023.675),
            (-1835.305, 4022.745),
            (-1834.96, 4022.725),
            (-1834.615, 4022.745),
        ],
        "4G11_2": [
            (-1835.295, 3243.605),
            (-1834.96, 3243.625),
            (-1834.625, 3243.605),
            (-1835.295, 3242.745),
            (-1834.96, 3242.725),
            (-1834.625, 3242.745),
        ],
        "4G11_3": [
            (-1835.285, 2463.535),
            (-1834.96, 2463.555),
            (-1834.635, 2463.535),
            (-1835.285, 2462.745),
            (-1834.96, 2462.725),
            (-1834.635, 2462.745),
        ],
        "4G21_1": [
            (2194.695, 6903.675),
            (2195.05, 6903.695),
            (2195.405, 6903.675),
            (2194.695, 6902.745),
            (2195.05, 6902.725),
            (2195.405, 6902.745),
        ],
        "4G21_2": [
            (2194.695, 5863.675),
            (2195.04, 5863.695),
            (2195.385, 5863.675),
            (2194.695, 5862.745),
            (2195.04, 5862.725),
            (2195.385, 5862.745),
        ],
        "4G22_1": [
            (2324.705, 6903.605),
            (2325.05, 6903.625),
            (2325.395, 6903.605),
            (2324.705, 6902.745),
            (2325.05, 6902.725),
            (2325.395, 6902.745),
        ],
        "4G22_2": [
            (2324.705, 5863.605),
            (2325.04, 5863.625),
            (2325.375, 5863.605),
            (2324.705, 5862.745),
            (2325.04, 5862.725),
            (2325.375, 5862.745),
        ],
        "4G23_1": [
            (2454.715, 6903.535),
            (2455.05, 6903.555),
            (2455.385, 6903.535),
            (2454.715, 6902.745),
            (2455.05, 6902.725),
            (2455.385, 6902.745),
        ],
        "4G23_2": [
            (2454.715, 5863.535),
            (2455.04, 5863.555),
            (2455.365, 5863.535),
            (2454.715, 5862.745),
            (2455.04, 5862.725),
            (2455.365, 5862.745),
        ],
        "7G11_1": [
            (-1315.625, 3830.905),
            (-1315.295, 3830.905),
            (-1314.96, 3830.925),
            (-1314.625, 3830.905),
            (-1315.625, 3829.745),
            (-1315.295, 3829.745),
            (-1314.96, 3829.725),
            (-1314.625, 3829.745),
            (-1314.295, 3829.745),
        ],
        "7G11_2": [
            (-1315.615, 2660.775),
            (-1315.285, 2660.775),
            (-1314.96, 2660.795),
            (-1314.635, 2660.775),
            (-1315.615, 2659.745),
            (-1315.285, 2659.745),
            (-1314.96, 2659.725),
            (-1314.635, 2659.745),
            (-1314.305, 2659.745),
        ],
        "8G11": [
            (-1185.615, 3764.855),
            (-1185.615, 3764.855),
            (-1185.285, 3764.855),
            (-1184.96, 3764.875),
            (-1184.635, 3764.855),
            (-1184.305, 3764.855),
            (-1185.615, 3763.745),
            (-1185.285, 3763.745),
            (-1184.96, 3763.725),
            (-1184.635, 3763.745),
            (-1184.305, 3763.745),
        ],
    }

    annotations = PolygonSet(layer=layer, datatype=datatype)
    vias = PolygonSet(layer=layer, datatype=datatype)

    for device in devices.keys():
        # print(device)
        coordinates = devices[device]
        annotations += Text(
            device,
            20,
            position=(coordinates[0][0], coordinates[0][1] + 0.5),
            layer=layer_annotation,
        )
        for via_coordinates in coordinates:
            vias += RectangleCentered(
                (via_coordinates[0], via_coordinates[1]),
                0.09,
                0.09,
                layer=layer,
                datatype=datatype,
            )

    total = vias + annotations
    return total


def dicing_saw_mark(substrate: str='si',
                    layer: int=1,
                    name: str='',
                    color: str='',
                    datatype: int=1,
                    ratio: float=5) -> PolygonSet:
    """
    Return dicing saw marks in a shape of a cross.
    Theses mark are done in such way that the blade thickness used at the BCAI
    will completely delete the mark from the leftover chips, a.k.a. the blade
    thickness corresponds to the mark width.
    Hence, for each substrate type corresponds a mark thickness:
        Si    -> 60um width
        Al2O3 -> 250um width

    Args:
        substrate: Nature of the substrate, must be: ('si', 'silicium', 'al2o3',
            'sapphire').
            Defaults to 'si'.
        layer: Number of the metal layer.
            Defaults to 0.
        name: Name of the metal layer.
            Defaults to ''.
        color: Color of the metal layer.
            Defaults to ''.
        datatype: Datatype of the metal layer.
            Defaults to 0.
        ratio: ratio length/width of the dicing saw mark.
            Defaults to 5.
    """

    if substrate.lower() in ('si', 'silicium'):
        w = 60
        l = ratio*w
    elif substrate.lower() in ('al2o3', 'sapphire'):
        w = 250
        l = ratio*w
    else:
        raise ValueError('substrate must be "si", or "al2o3"')

    s = l/2 - w/2

    t = PolygonSet(polygons   = [[(-l/2, w/2)]],
                   layers     = [layer],
                   datatypes  = [datatype],
                   colors     = [color],
                   names      = [name])
    t > ( s,  0)
    t > ( 0,  s)
    t > ( w,  0)
    t > ( 0, -s)
    t > ( s,  0)
    t > ( 0, -w)
    t > (-s,  0)
    t > ( 0, -s)
    t > (-w,  0)
    t > ( 0,  s)
    t > (-s,  0)
    t > ( 0,  w/2)

    return t
