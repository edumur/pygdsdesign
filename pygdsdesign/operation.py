from math import ceil
from typing import Any, Dict, List, Literal, Tuple, Optional
import warnings
import copy as libcopy
import numpy as np

import clipper
from pygdsdesign.polygons import Rectangle
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.typing_local import Coordinates


def boolean(operand1: PolygonSet,
            operand2: PolygonSet|list,
            operation: Literal['or', 'and', 'xor', 'not'],
            precision: float=0.001,
            layer: int=0,
            datatype: int=0,
            name: str ='',
            color: str ='') -> Optional[PolygonSet]:
    """
    Execute any boolean operation between 2 polygons or polygon sets.

    Parameters
    ----------
    operand1 : `PolygonSet`, `CellReference`, `CellArray` or iterable
        First operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operand2 : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Second operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operation : {'or', 'and', 'xor', 'not'}
        Boolean operation to be executed.  The 'not' operation returns
        the difference ``operand1 - operand2``.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : PolygonSet or None
        Result of the boolean operation.
    """
    poly1 = _gather_polys(operand1)
    poly2 = _gather_polys(operand2)

    if len(poly2) == 0:
        if operation == "not" or operation=="xor":
            if len(poly1) == 0:
                warnings.warn("[pygdsdesign] You try to 'boolean' on a empty polygonSet",stacklevel=4,)
                return None
            return PolygonSet(poly1, operand1.layers, operand1.datatypes, operand1.names, operand1.colors)
        poly2.append(poly1.pop())

    result = clipper.clip(poly1, poly2, operation, 1 / precision)

    if len(result) == 0:
        warnings.warn("[pygdsdesign] The result of your boolean operation is None",stacklevel=4,)
        return None

    return PolygonSet(result, [layer]*len(result), [datatype]*len(result), [name]*len(result), [color]*len(result))


def addition(operand1: PolygonSet,
             operand2: PolygonSet,
             precision: float=0.001,
             layer: int=0,
             datatype: int=0,
             name: str ='',
             color: str ='') -> Optional[PolygonSet]:
    """
    Execute the OR boolean operation between 2 polygon sets.

    Parameters
    ----------
    operand1 : `PolygonSet`, `CellReference`, `CellArray` or iterable
        First operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operand2 : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Second operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : PolygonSet or None
        Result of the boolean operation.
    """
    return boolean(operand1, operand2, "or", precision, layer, datatype, name, color)


def subtraction(operand1: PolygonSet,
                 operand2: PolygonSet,
                 precision: float=0.001,
                 layer: int=0,
                 datatype: int=0,
                 name: str ='',
                 color: str ='') -> Optional[PolygonSet]:
    """
    Execute the NOT boolean operation between 2 polygon sets.

    Parameters
    ----------
    operand1 : `PolygonSet`, `CellReference`, `CellArray` or iterable
        First operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operand2 : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Second operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : PolygonSet or None
        Result of the boolean operation.
    """
    return boolean(operand1, operand2, "not", precision, layer, datatype, name, color)


def intersection(operand1: PolygonSet,
                 operand2: PolygonSet,
                 precision: float=0.001,
                 layer: int=0,
                 datatype: int=0,
                 name: str ='',
                 color: str ='') -> Optional[PolygonSet]:
    """
    Execute the AND boolean operation between 2 polygon sets.

    Parameters
    ----------
    operand1 : `PolygonSet`, `CellReference`, `CellArray` or iterable
        First operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operand2 : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Second operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : PolygonSet or None
        Result of the boolean operation.
    """
    return boolean(operand1, operand2, "and", precision, layer, datatype, name, color)


def difference(operand1: PolygonSet,
               operand2: PolygonSet,
               precision: float=0.001,
               layer: int=0,
               datatype: int=0,
               name: str ='',
               color: str ='') -> Optional[PolygonSet]:
    """
    Execute the XOR boolean operation between 2 polygon sets.

    Parameters
    ----------
    operand1 : `PolygonSet`, `CellReference`, `CellArray` or iterable
        First operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    operand2 : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Second operand.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : PolygonSet or None
        Result of the boolean operation.
    """
    return boolean(operand1, operand2, "xor", precision, layer, datatype, name, color)


def crop(
    polygons: PolygonSet,
    orientation: Literal["top", "right", "bottom", "left"],
    value: float,
    layer: int = 0,
    datatype: int = 0,
    name: str = "",
    color: str = "",
) -> Optional[PolygonSet]:
    """
    Crop a polygon or a collection of polygon by a certain value from a certain
    direction

    Args:
        orientation: the orientation of the crop.
            Must be 'top', 'right', 'bottom' or 'left'.
        value: value of the crop in um
    """
    bounding_box = polygons.get_bounding_box()

    if bounding_box is None:
        warnings.warn("[pygdsdesign] Can't crop a polygon with a null area.")
        return None

    ((x0, y0), (x1, y1)) = bounding_box

    if orientation=='top':
        t = Rectangle((x0, y0), (x1, y1-value))
    elif orientation=='right':
        t = Rectangle((x0, y0), (x1-value, y1))
    elif orientation=='bottom':
        t = Rectangle((x0, y0+value), (x1, y1))
    elif orientation=='left':
        t = Rectangle((x0+value, y0), (x1, y1))
    else:
        raise ValueError("orientation must be 'top', 'right', 'bottom' or 'left'")

    return boolean(polygons, t, 'and', layer=layer, datatype=datatype, name=name, color=color)


def merge(polygons) -> PolygonSet:
    """
    Merge polygons of the same layer and datatype
    """
    # Sort all the polygons per layer
    layers: Dict[Tuple[int, int, str, str], List[np.ndarray]] = {}

    for poly, layer, datatype, name, color in zip(polygons.polygons, polygons.layers, polygons.datatypes, polygons.names, polygons.colors):
        if (layer, datatype, name, color) in layers:
            layers[(layer, datatype, name, color)].append(poly)
        else:
            layers[(layer, datatype, name, color)] = [poly]

    # Merge all polygons layer per layer
    tot = PolygonSet(
        polygons=[np.array([[0.0, 0.0]])],
        layers=[layer],
        datatypes=[datatype],
        names=[name],
        colors=[color],
    )

    for (layer, datatype, name, color), poly in layers.items():
        temp = boolean(poly, [],
                       'or',
                       layer=layer,
                       datatype=datatype,
                       name=name,
                       color=color)
        if temp is not None:
            tot += temp

    return tot


def _gather_polys(args):
    """
    Gather polygons from different argument types into a list.

    Parameters
    ----------
    args : None, `PolygonSet`, `CellReference`, `CellArray` or iterable
        Polygon types.  If this is an iterable, each element must be a
        `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.

    Returns
    -------
    out : list of numpy array[N][2]
        List of polygons.
    """
    if args is None:
        warnings.warn("[pygdsdesign] You try to '_gather_polys' on empty polygonSets",stacklevel=4)
        return []

    if isinstance(args, PolygonSet):
        return [p for p in args.polygons]

    polys = []

    for p in args:
        if isinstance(p, PolygonSet):
            polys.extend(p.polygons)
        else:
            polys.append(p)

    return polys


def slice(polygons, position, axis, precision=1e-3, layer=0, datatype=0):
    """
    Slice polygons and polygon sets at given positions along an axis.

    Parameters
    ----------
    polygons : `PolygonSet`, `CellReference`, `CellArray` or iterable
        Operand of the slice operation.  If this is an iterable, each
        element must be a `PolygonSet`, `CellReference`, `CellArray`,
        or an array-like[N][2] of vertices of a polygon.
    position : number or list of numbers
        Positions to perform the slicing operation along the specified
        axis.
    axis : 0 or 1
        Axis along which the polygon will be sliced.
    precision : float
        Desired precision for rounding vertice coordinates.
    layer : integer, list
        The GDSII layer numbers for the elements between each division.
        If the number of layers in the list is less than the number of
        divided regions, the list is repeated.
    datatype : integer, list
        The GDSII datatype for the resulting element (between 0 and
        255).  If the number of datatypes in the list is less than the
        number of divided regions, the list is repeated.

    Returns
    -------
    out : list[N] of `PolygonSet` or None
        Result of the slicing operation, with N = len(positions) + 1.
        Each PolygonSet comprises all polygons between 2 adjacent
        slicing positions, in crescent order.

    Examples
    --------
    >>> ring = Round((0, 0), 10, inner_radius = 5)
    >>> result = slice(ring, [-7, 7], 0)
    >>> cell.add(result[1])
    """
    polys = _gather_polys(polygons)
    if not isinstance(layer, list):
        layer = [layer]
    if not isinstance(datatype, list):
        datatype = [datatype]
    if not isinstance(position, list):
        pos = [position]
    else:
        pos = sorted(position)

    result = [[] for _ in range(len(pos) + 1)]
    scaling = 1 / precision

    for pol in polys:
        for r, p in zip(result, clipper._chop(pol, pos, axis, scaling)):
            r.extend(p)

    for i in range(len(result)):
        if len(result[i]) == 0:
            result[i] = None
        else:
            result[i] = PolygonSet(
                result[i], layer[i % len(layer)], datatype[i % len(datatype)]
            )

    return result


def offset(polygons: PolygonSet,
           distance: float,
           join: str='miter',
           tolerance: float=2,
           precision: float=0.001,
           join_first=False,
           layer: int=0,
           datatype: int=0,
           name: str='',
           color: str='') -> Optional[PolygonSet]:
    """
    Shrink or expand a polygon set.

    Parameters
    ----------
    polygons : `PolygonSet`, `CellReference`, `CellArray` or iterable
        Polygons to be offset.  If this is an iterable, each element
        must be a `PolygonSet`, `CellReference`, `CellArray`, or an
        array-like[N][2] of vertices of a polygon.
    distance : number
        Offset distance.  Positive to expand, negative to shrink.
    join : 'miter', 'bevel', 'round'
        Type of join used to create the offset polygon.
    tolerance : number
        For miter joints, this number must be at least 2 and it
        represents the maximal distance in multiples of offset between
        new vertices and their original position before beveling to
        avoid spikes at acute joints.  For round joints, it indicates
        the curvature resolution in number of points per full circle.
    precision : float
        Desired precision for rounding vertex coordinates.
    join_first : bool
        Join all paths before offsetting to avoid unnecessary joins in
        adjacent polygon sides.
    layer : integer
        The GDSII layer number for the resulting element.
    datatype : integer
        The GDSII datatype for the resulting element (between 0 and
        255).

    Returns
    -------
    out : `PolygonSet` or None
        Return the offset shape as a set of polygons.
    """
    result = clipper.offset(
        _gather_polys(polygons),
        distance,
        join,
        tolerance,
        1 / precision,
        1 if join_first else 0,
    )

    if len(result) == 0:
        warnings.warn("[pygdsdesign] The result of 'offset' is None",stacklevel=4,)
        return None

    return PolygonSet(result, [layer]*len(result), [datatype]*len(result), [name]*len(result), [color]*len(result))


def grid_cover(polygons: PolygonSet,
               square_width: float=10,
               square_gap: float=12,
               safety_margin: float=10,
               centered: bool=False,
               noise: float=0,
               only_square: bool=True,
               layer: int=1,
               datatype: int=0,
               name: str='',
               color: str ='') ->  PolygonSet:
    """
        Create a grid pattern of squares which follows any shape given in polygons
        after shrinking it by a safety margin.
        The square size and gap between them are given as free parameters.

        Args:
            polygons: shape from which the grid pattern is built upon.
            square_width: Width of the square in um.
                Defaults to 10um.
            square_gap: Space between the square in um.
                Defaults to 12um.
            safety_margin: shrinking scaling distance taken from the polygon in um.
                Defaults to 10um.
            centered: if False, the grid bottom left corresponds to the polygons
                bottom left.
                If True, the grid center corresponds to the polygon center.
                Defaults to False.
            noise: Noise parameter in um to generate random square positions.
                Each square is translated by a random amount given by the noise
                parameter. A noise value uf 4 um will give a random shift
                from -4 um to +4 um in x and y.
                Defaults to 0, resulting in a regular grid.
            only_square: Remove all polygons that are not stricly
                `square_width` x `square_width` shape.
                Defaults to True.
            layer: gds layer of the grid cover.
                Defaults to 1.
            datatype: gds datatype of the grid cover.
                Defaults to 1.
            name: gds name of the grid cover.
                Defaults to ''.
            color: gds color of the grid cover.
                Defaults to ''.
    """
    poly = offset(polygons=polygons,
                  distance=-safety_margin,
                  join_first=True,
                  layer=layer,
                  datatype=datatype,
                  name=name,
                  color=color)

    if poly is not None:

        # First we merge the polygons and then we iterate over it
        # That way avoid to create grid in between polygons
        poly = merge(poly)

        resultPoly = PolygonSet()

        for p in poly.polygons:
                        # skip empty polygon
            if (p==[[0,0]]).all():
                continue

            # create holed framed
            dx = np.ptp(p[:, 0])
            dy = np.ptp(p[:, 1])

            # TODO check if it's not needed to add 1 here
            nb_square_x = ceil(dx/(square_width+square_gap))
            nb_square_y = ceil(dy/(square_width+square_gap))

            temp = np.array([(0, 0),
                             (0, square_width),
                             (square_width, square_width),
                             (square_width, 0)], dtype=np.float64)

            xi = 1

            while xi<nb_square_x+1:
                temp = np.concatenate((temp, temp+np.array([temp[:,0].max()-temp[:,0].min()+square_gap, 0])))
                xi = xi*2

            xi = 1

            while xi<nb_square_y+1:
                temp = np.concatenate((temp, temp+np.array([0, temp[:,1].max()-temp[:,1].min()+square_gap])))
                xi = xi*2

            # If user wants to add some noise to the square positions
            if noise!=0:
                t = 2*noise * np.random.random((int(temp.shape[0]/4), 2)) - noise
                temp += np.repeat(t, 4, axis=0)

            # If use wants to center the whole grid
            if centered:
                temp += np.array([-(temp[:,0].max()+temp[:,0].min())/2, -(temp[:,1].max()+temp[:,1].min())/2])
                temp += np.array([(p[:,0].max()+p[:,0].min())/2, (p[:,1].max()+p[:,1].min())/2])
            else:
                temp += np.array([p[:,0].min(), p[:,1].min()])


            # Boolean operation
            polys = clipper.clip(np.split(temp, int(len(temp[:,0])/4)),
                        [p],
                        'and',
                        1000)

            # if we still have some square left
            if len(polys) >0:
                resultPoly += PolygonSet(polys)
    else:
        resultPoly = PolygonSet()

    # If user wants to keep only entire squares
    if only_square:
        temp = []
        for p in resultPoly.polygons:
            # We keep only quadrilateral
            if p.shape[0]==4 and p.shape[1]==2:
                # We keep only square
                if (    abs(p[1,0]-p[0,0])==square_width
                    and abs(p[3,1]-p[0,1])==square_width):
                    temp.append(p)
        resultPoly = PolygonSet(temp)

    return resultPoly.change_layer(layer=layer,
                                   datatype=datatype,
                                   name=name,
                                   color=color)


def inverse_polarity(polygons: PolygonSet,
                     safety_marge:float=1e-3,
                     layer: int=1,
                     datatype: int=0,
                     name: str='',
                     color: str ='') -> Optional[PolygonSet]:
    """
    Inverse the polarity of the polygons.
    To do so a marge of 1nm is added to the 4 most extreme points.

    Parameters
    -----------
    safety_marge : float
        Marge added for the polarity inversion.
    """
    bounding_box = polygons.get_bounding_box()

    if bounding_box is None:
        warnings.warn("[pygdsdesign] Can't inverse_polarity a polygon with a null area.")
        return None
    a, b = bounding_box
    r = Rectangle((a[0]-safety_marge, a[1]-safety_marge),
                  (b[0]+safety_marge, b[1]+safety_marge))
    poly1 = r.polygons
    poly2 = polygons.polygons
    result = clipper.clip(poly1, poly2, "xor", 1 / 0.001)

    if len(result) == 0:
        warnings.warn("[pygdsdesign] The result of 'inverse_polarity' is None",stacklevel=4,)
        return None

    return PolygonSet(result, [layer]*len(result), [datatype]*len(result), [name]*len(result), [color]*len(result))


def inside(points: Coordinates,
           polygons: PolygonSet,
           short_circuit: Literal['any', 'all']='any',
           precision: float=0.001) -> Tuple[bool]:
    """
    Test whether each of the points is within the given set of polygons.

    Parameters
    ----------
    points : array-like[N][2] or sequence of array-like[N][2]
        Coordinates of the points to be tested or groups of points to be
        tested together.
    polygons : `PolygonSet`, `CellReference`, `CellArray` or iterable
        Polygons to be tested against.  If this is an iterable, each
        element must be a `PolygonSet`, `CellReference`, `CellArray`,
        or an array-like[N][2] of vertices of a polygon.
    short_circuit : {'any', 'all'}
        If `points` is a sequence of point groups, testing within each
        group will be short-circuited if any of the points in the group
        is inside ('any') or outside ('all') the polygons.  If `points`
        is simply a sequence of points, this parameter has no effect.
    precision : float
        Desired precision for rounding vertice coordinates.

    Returns
    -------
    out : tuple
        Tuple of booleans indicating if each of the points or point
        groups is inside the set of polygons.
    """
    polys = _gather_polys(polygons)
    pts: Tuple[List[np.ndarray]]|Coordinates

    if np.isscalar(points[0][0]):
        pts = (points,)
        sc = 0
    else:
        pts = points
        sc = 1 if short_circuit == "any" else -1

    return clipper.inside(pts, polys, sc, 1 / precision)


def copy(obj: Any,
         dx:float=0,
         dy:float=0) -> Any:
    """
    Create a copy of `obj` and translate it by (dx, dy).

    Parameters
    ----------
    obj : translatable object
        Object to be copied.
    dx : number
        Distance to move in the x-direction.
    dy : number
        Distance to move in the y-direction.


    Returns
    -------
    out : translatable object
        Translated copy of original `obj`

    Examples
    --------
    >>> rectangle = Rectangle((0, 0), (10, 20))
    >>> rectangle2 = copy(rectangle, 2,0)
    >>> myCell.add(rectangle)
    >>> myCell.add(rectangle2)
    """

    newObj = libcopy.deepcopy(obj)
    if dx != 0 or dy != 0:
        newObj.translate(dx, dy)
    return newObj


def select_polygon_per_layer(polygons: PolygonSet,
                             layer: int,
                             datatype: int=0,
                             merging: bool=False) -> PolygonSet:
    """
    Return a copy of the polygon(s) corresponding to the given layer and
    datatype.
    By default the datatype is 0.
    This method does not change the original Polygon or PolygonSet.

    Args:
        layer : Layer number.
        datatype: gds datatype of the grid cover.
            Defaults to 0.
        merging : If merging is True, merge before returning.
            Defaults to False.
    """

    # We create a numpy mask of the single layer we want to keep
    ls = np.array(polygons.layers)
    mask1 = ls==layer

    # We create a numpy mask of the single datatype we want to keep
    ds = np.array(polygons.datatypes)
    mask2 = ds==datatype

    # We combine both mask with an "and" boolean
    mask = mask1*mask2

    # We create an empty Polygon
    tot = PolygonSet()

    # We fill that polygon with the mask
    tot.polygons  = list(np.array(polygons.polygons, dtype=object)[mask])
    tot.layers    = list(ls[mask])
    tot.datatypes = list(ds[mask])
    tot.colors    = list(np.array(polygons.colors)[mask])
    tot.names     = list(np.array(polygons.names)[mask])

    if merging:
        return merge(tot)
    else:
        return tot


def select_polygon_per_name(polygons: PolygonSet,
                            name: int,
                            merging: bool=False) -> PolygonSet:
    """
    Return a copy of the polygon(s) corresponding to the given name
    This method does not change the original Polygon or PolygonSet.

    Args:
        layer : Layer number.
        datatype: gds datatype of the grid cover.
            Defaults to 0.
        merging : If merging is True, merge before returning.
            Defaults to False.
    """

    # We create a numpy mask of the single layer we want to keep
    ns = np.array(polygons.names)
    mask = ns==name

    # We create an empty Polygon
    tot = PolygonSet()

    # We fill that polygon with the mask
    tot.polygons  = list(np.array(polygons.polygons, dtype=object)[mask])
    tot.layers    = list(np.array(polygons.layers)[mask])
    tot.datatypes = list(np.array(polygons.datatypes)[mask])
    tot.colors    = list(np.array(polygons.colors)[mask])
    tot.names     = list(ns)

    if merging:
        return merge(tot)
    else:
        return tot
