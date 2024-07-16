from typing import List, Literal, Optional, Dict, Tuple
from typing_extensions import Self
import numpy as np
import struct
import warnings
from itertools import chain
_mpone = np.array((-1.0, 1.0))
from io import IOBase


from pygdsdesign.typing_local import Coordinate, Coordinates


class PolygonSet():

    __slots__ = "layers", "datatypes", "polygons", "properties", "names", "colors"

    def __init__(self, polygons: Coordinates=None,
                       layers: List[int]=None,
                       datatypes: List[int]=None,
                       names: List[str]=None,
                       colors: List[str]=None,
                       **kwargs) -> None:

        # Getting rid of the mutability of python's default arguments
        if polygons is None:
            polygons = [np.array([[0., 0.]])]
        if layers is None:
            layers = [0]
        if datatypes is None:
            datatypes = [0]
        if names is None:
            names = ['']
        if colors is None:
            colors = ['']

        self.polygons: List[np.ndarray] = [np.asarray(p) for p in polygons]
        self.layers: List[int] = layers
        self.datatypes: List[int] = datatypes
        self.names: List[str]  = names
        self.colors: List[str] = colors
        self.properties = {}

        # to keep retrop-compatibility with "layer" old attribute
        # To allow wasy use of layer dictionnary
        if 'layer' in kwargs:
            self.layers = [kwargs['layer']]

        # Find the maximum array length
        max_length = max(len(self.polygons), len(self.layers), len(self.datatypes), len(self.names), len(self.colors))

        # Equalize the size of each argument
        for arg, last_element in [(self.polygons, self.polygons[-1]),
                                    (self.layers, self.layers[-1]),
                                    (self.datatypes, self.datatypes[-1]),
                                    (self.names, self.names[-1]),
                                    (self.colors, self.colors[-1])]:
            while len(arg) < max_length:
                arg.append(last_element)


    def __gt__(self, point: Coordinate) -> Self:
        """
        Add a vertice to the polygon in respect to the last vertice coordinate

        Parameters
        ----------
        point : tuple
            Point coordinate (x, y)
        """

        self.polygons[0] = np.vstack((self.polygons[0], self.polygons[0][-1]+np.asarray(point)))

        return self


    def __add__(self, polygon: Self) -> Self:
        """
        Add two polygons together

        Parameters
        -----------
        polygon : Polygon
            Polygon to be added
        """

        ### Carreful with mutability here!

        # Check if empty polygon -> we replace it by the non-empty one
        if len(self.polygons)==1:
            if (self.polygons[0]==np.array([[0,0]])).all():

                self.polygons = polygon.polygons
                self.layers = polygon.layers
                self.datatypes = polygon.datatypes
                self.names = polygon.names
                self.colors = polygon.colors

                return self

        self.polygons = self.polygons + polygon.polygons
        self.layers = self.layers + polygon.layers
        self.datatypes = self.datatypes + polygon.datatypes
        self.names = self.names + polygon.names
        self.colors = self.colors + polygon.colors

        return self


    def __str__(self) -> str:
        return (
            "PolygonSet ({} polygons, {} vertices, layers {}, datatypes {}, names {}, colors {})"
        ).format(
            len(self.polygons),
            sum([len(p) for p in self.polygons]),
            self.layers,
            self.datatypes,
            self.names,
            self.colors,
        )


    def get_area(self, by_spec=False) -> float | Dict[Tuple[int, int], float]:
        """
        Calculate the total area of this polygon set.

        Parameters
        ----------
        by_spec : bool
            If True, the return value is a dictionary with
            ``{(layer, datatype): area}``.

        Returns
        -------
        out : number, dictionary
            Area of this object.
        """
        if by_spec:
            path_area = {}
            for poly, key in zip(self.polygons, zip(self.layers, self.datatypes)):
                poly_area = 0
                for ii in range(1, len(poly) - 1):
                    poly_area += (poly[0][0] - poly[ii + 1][0]) * (
                        poly[ii][1] - poly[0][1]
                    ) - (poly[0][1] - poly[ii + 1][1]) * (poly[ii][0] - poly[0][0])
                if key in path_area:
                    path_area[key] += 0.5 * abs(poly_area)
                else:
                    path_area[key] = 0.5 * abs(poly_area)
        else:
            path_area = 0.
            for points in self.polygons:
                poly_area = 0
                point0 = points[0][0]
                point01 = points[0][1]
                for ii, point in enumerate(points[1:-1], start=1):
                    poly_area += (point0 - points[ii + 1][0]) * (point[1] - point01) - (point01 - points[ii + 1][1]) * (point[0] - point0)
                path_area += 0.5 * abs(poly_area)
        return path_area


    def get_center(self) -> Coordinate:
        """
        Return the center of the polygon.
        """

        a, b = self.get_bounding_box()

        return (a[0]+b[0])/2., (a[1]+b[1])/2.


    def get_bounding_box(self) -> Optional[Tuple[Coordinate, Coordinate]]:
        """
        Calculate the bounding box of the polygons.

        Returns
        -------
        out : Numpy array[2, 2] or None
            Bounding box of this polygon in the form [[x_min, y_min],
            [x_max, y_max]], or None if the polygon is empty.
        """
        if len(self.polygons) == 0:
            warnings.warn("[pygdsdesign] You try to 'get_bounding_box' on a empty polygonSet",stacklevel=4,)
            return None

        # Convert the polygons into a single 3D numpy array
        all_points = np.vstack(self.polygons)

        min_x = np.min(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_x = np.max(all_points[:, 0])
        max_y = np.max(all_points[:, 1])

        return np.array([[min_x, min_y], [max_x, max_y]])


    def get_size(self) -> Coordinate:
        """
        Return the size the polygon as length along (x, y).
        """

        a, b = self.get_bounding_box()
        return b[0]-a[0],  b[1]-a[1]


    def rotate(self, angle: float,
                     center: Coordinate=(0, 0)) -> Self:
        """
        Rotate this object.

        Parameters
        ----------
        angle : number
            The angle of rotation (in *radians*).
        center : array-like[2]
            Center point for the rotation.

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        ca = np.cos(angle)
        sa = np.sin(angle) * _mpone
        c0 = np.array(center)
        new_polys = []
         # Tester avec  self.polygons = [((points - c0) * ca + (points - c0)[:, ::-1] * sa + c0) for points in self.polygons]

        for points in self.polygons:
            pts = points - c0
            new_polys.append(pts * ca + pts[:, ::-1] * sa + c0)
        self.polygons = new_polys

        return self


    def remove_polygon_from_layer(self, layer: int) -> Self:
        """
        Remove all polygons of the given layer from the current PolygonSet.
        For instance, if the current PolygonSet contains Polygon with layer
        (1, 2, 3, 6, 9), using this method with `layer=3` as input parameter result
        in a PolygonSet containing Polygon with layer (1, 2, 6, 9).

        Args:
            layer : Layer number.
        """

        # We create a numpy mask of the single layer we want to keep
        ls = np.array(self.layers)
        mask = ls!=layer

        # We filter the array with the mask
        self.polygons  = list(np.array(self.polygons, dtype=object)[mask])
        self.layers    = list(ls[mask])
        self.datatypes = list(np.array(self.datatypes)[mask])
        self.colors    = list(np.array(self.colors)[mask])
        self.names     = list(np.array(self.names)[mask])

        return self


    def scale(self, scalex: float,
                    scaley: Optional[float]=None,
                    center: Tuple[float, float]=(0., 0.)) -> Self:
        """
        Scale this object.

        Parameters
        ----------
        scalex : number
            Scaling factor along the first axis.
        scaley : number or None
            Scaling factor along the second axis.  If None, same as
            `scalex`.
        center : array-like[2]
            Center point for the scaling operation.

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        c0 = np.array(center)
        s = scalex if scaley is None else np.array((scalex, scaley))
        self.polygons = [(points - c0) * s + c0 for points in self.polygons]
        return self


    def center(self) -> Self:
        """
        Centered the polygon to (0, 0).
        """

        x0, y0 = self.get_center()
        self.translate(-x0, -y0)

        return self


    def flip(self, axis: Literal['x', 'y']) -> Self:
        """
        Flip the polygon along its central x or y axis.

        Parameters
        ----------
        axis : float
            Central axis of the flip.
        """

        x0, y0 = self.get_center()
        self.translate(-x0, -y0)

        if axis.lower()=='x':
            self.polygons = [np.vstack((p[:, 0], p[:, 1]*-1)).T for p in self.polygons]
        elif axis.lower()=='y':
            self.polygons = [np.vstack((p[:, 0]*-1, p[:, 1])).T for p in self.polygons]
        else:
            raise ValueError('axis must be "x", "y"')

        self.translate(x0, y0)

        return self


    def translate(self, dx: float,
                        dy: float) -> 'PolygonSet':
        """
        Translate this polygon.

        Parameters
        ----------
        dx : number
            Distance to move in the x-direction.
        dy : number
            Distance to move in the y-direction.

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        vec = np.array((dx, dy))
        self.polygons = [points + vec for points in self.polygons]
        return self


    def to_gds(self, outfile: IOBase,
                     multiplier: float) -> None:
            """
            Convert this object to a series of GDSII elements.

            Parameters
            ----------
            outfile : open file
                Output to write the GDSII.
            multiplier : number
                A number that multiplies all dimensions written in the GDSII
                elements.
            """
            for ii, polygon in enumerate(self.polygons):
                if len(polygon) > 8190:
                    # warnings.warn(
                    #     "[pygdsdesign] Polygons with more than 8190 are not supported by the "
                    #     "official GDSII specification.  This GDSII file might not be "
                    #     "compatible with all readers.",
                    #     stacklevel=4,
                    # )
                    outfile.write(
                        struct.pack(
                            ">4Hh2Hh",
                            4,
                            0x0800,
                            6,
                            0x0D02,
                            self.layers[ii],
                            6,
                            0x0E02,
                            self.datatypes[ii],
                        )
                    )
                    xy = np.empty((polygon.shape[0] + 1, 2), dtype=">i4")
                    xy[:-1] = np.round(polygon * multiplier)
                    xy[-1] = xy[0]
                    i0 = 0
                    while i0 < xy.shape[0]:
                        i1 = min(i0 + 8190, xy.shape[0])
                        outfile.write(struct.pack(">2H", 4 + 8 * (i1 - i0), 0x1003))
                        outfile.write(xy[i0:i1].tobytes())
                        i0 = i1
                else:
                    outfile.write(
                        struct.pack(
                            ">4Hh2Hh2H",
                            4,
                            0x0800,
                            6,
                            0x0D02,
                            self.layers[ii],
                            6,
                            0x0E02,
                            self.datatypes[ii],
                            12 + 8 * len(polygon),
                            0x1003,
                        )
                    )
                    # TODO -> enlever le round() ?
                    xy = np.round(polygon * multiplier).astype(">i4")
                    outfile.write(xy.tobytes())
                    outfile.write(xy[0].tobytes())
                if self.properties is not None and len(self.properties) > 0:
                    size = 0
                    for attr, value in self.properties.items():
                        if len(value) % 2 != 0:
                            value = value + "\0"
                        outfile.write(
                            struct.pack(">5H", 6, 0x2B02, attr, 4 + len(value), 0x2C06)
                        )
                        outfile.write(value.encode("ascii"))
                        size += len(value) + 2
                    if size > 128:
                        warnings.warn(
                            "[pygdsdesign] Properties with size larger than 128 bytes are not "
                            "officially supported by the GDSII specification.  This file "
                            "might not be compatible with all readers.",
                            stacklevel=4,
                        )
                outfile.write(struct.pack(">2H", 4, 0x1100))


    def mirror(self, p1: Tuple[float, float],
                     p2: Tuple[float, float]=(0, 0)) -> Self:
        """
        Mirror the polygons over a line through points 1 and 2

        Parameters
        ----------
        p1 : array-like[2]
            first point defining the reflection line
        p2 : array-like[2]
            second point defining the reflection line

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        origin = np.array(p1)
        vec = np.array(p2) - origin
        vec_r = vec * (2 / np.inner(vec, vec))
        self.polygons = [
            np.outer(np.inner(points - origin, vec_r), vec) - points + 2 * origin
            for points in self.polygons
        ]
        return self


    def fracture(self, max_points: int=199,
                       precision: float=1e-3) -> Self:
        """
        Slice these polygons in the horizontal and vertical directions
        so that each resulting piece has at most `max_points`.  This
        operation occurs in place.

        Parameters
        ----------
        max_points : integer
            Maximal number of points in each resulting polygon (at least
            5 for the fracture to occur).
        precision : float
            Desired precision for rounding vertice coordinates.

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        if max_points > 4:
            ii = 0
            while ii < len(self.polygons):
                if len(self.polygons[ii]) > max_points:
                    pts0 = sorted(self.polygons[ii][:, 0])
                    pts1 = sorted(self.polygons[ii][:, 1])
                    ncuts = len(pts0) // max_points
                    if pts0[-1] - pts0[0] > pts1[-1] - pts1[0]:
                        # Vertical cuts
                        cuts = [
                            pts0[int(i * len(pts0) / (ncuts + 1.0) + 0.5)]
                            for i in range(1, ncuts + 1)
                        ]
                        chopped = clipper._chop(
                            self.polygons[ii], cuts, 0, 1 / precision
                        )
                    else:
                        # Horizontal cuts
                        cuts = [
                            pts1[int(i * len(pts1) / (ncuts + 1.0) + 0.5)]
                            for i in range(1, ncuts + 1)
                        ]
                        chopped = clipper._chop(
                            self.polygons[ii], cuts, 1, 1 / precision
                        )
                    self.polygons.pop(ii)
                    layer = self.layers.pop(ii)
                    datatype = self.datatypes.pop(ii)
                    color = self.colors.pop(ii)
                    name = self.names.pop(ii)
                    self.polygons.extend(
                        np.array(x) for x in chain.from_iterable(chopped)
                    )
                    npols = sum(len(c) for c in chopped)
                    self.layers.extend(layer for _ in range(npols))
                    self.datatypes.extend(datatype for _ in range(npols))
                    self.colors.extend(color for _ in range(npols))
                    self.names.extend(name for _ in range(npols))
                else:
                    ii += 1
        return self


    def fillet(self, radius: float|np.ndarray|list,
                     points_per_2pi: int=128) -> Self:
        """
        Round the corners of these polygons and fractures them into
        polygons with less vertices if necessary.

        Parameters
        ----------
        radius : number, array-like
            Radius of the corners.  If number: all corners filleted by
            that amount.  If array: specify fillet radii on a
            per-polygon basis (length must be equal to the number of
            polygons in this `PolygonSet`).  Each element in the array
            can be a number (all corners filleted by the same amount) or
            another array of numbers, one per polygon vertex.
            Alternatively, the array can be flattened to have one radius
            per `PolygonSet` vertex.
        points_per_2pi : integer
            Number of vertices used to approximate a full circle.  The
            number of vertices in each corner of the polygon will be the
            fraction of this number corresponding to the angle
            encompassed by that corner with respect to 2 pi.

        Returns
        -------
        out : `PolygonSet`
            This object.
        """
        two_pi = 2 * np.pi

        if np.isscalar(radius):
            radii = [[radius] * p.shape[0] for p in self.polygons]
        else:
            assert isinstance(radius, np.ndarray)
            if len(radius) == len(self.polygons):
                radii = []
                for r, p in zip(radius, self.polygons):
                    if np.isscalar(r):
                        radii.append([r] * p.shape[0])
                    else:
                        if len(r) != p.shape[0]:
                            raise ValueError(
                                "[GDSPY] Wrong length in fillet radius list.  "
                                "Found {} radii for polygon with {} vertices.".format(
                                    len(r), len(p.shape[0])
                                )
                            )
                        radii.append(r)
            else:
                total = sum(p.shape[0] for p in self.polygons)
                if len(radius) != total:
                    raise ValueError(
                        "[GDSPY] Wrong length in fillet radius list.  "
                        "Expected lengths are {} or {}; got {}.".format(
                            len(self.polygons), total, len(radius)
                        )
                    )
                radii = []
                n = 0
                for p in self.polygons:
                    radii.append(radius[n : n + p.shape[0]])
                    n += p.shape[0]

        for jj in range(len(self.polygons)):
            vec = self.polygons[jj].astype(float) - np.roll(self.polygons[jj], 1, 0)
            length = (vec[:, 0] ** 2 + vec[:, 1] ** 2) ** 0.5
            ii = np.flatnonzero(length)
            if len(ii) < len(length):
                self.polygons[jj] = np.array(self.polygons[jj][ii])
                radii[jj] = [radii[jj][i] for i in ii]
                vec = self.polygons[jj].astype(float) - np.roll(
                    self.polygons[jj], 1, 0
                )
                length = (vec[:, 0] ** 2 + vec[:, 1] ** 2) ** 0.5
            vec[:, 0] = vec[:, 0] / length
            vec[:, 1] = vec[:, 1] / length
            dvec = np.roll(vec, -1, 0) - vec
            norm = (dvec[:, 0] ** 2 + dvec[:, 1] ** 2) ** 0.5
            ii = np.flatnonzero(norm)
            dvec[ii, 0] = dvec[ii, 0] / norm[ii]
            dvec[ii, 1] = dvec[ii, 1] / norm[ii]
            dot = np.roll(vec, -1, 0) * vec
            theta = np.arccos(dot[:, 0] + dot[:, 1])
            ct = np.cos(theta * 0.5)
            tt = np.tan(theta * 0.5)

            new_points = []
            for ii in range(-1, len(self.polygons[jj]) - 1):
                if theta[ii] > 1e-6:
                    a0 = -vec[ii] * tt[ii] - dvec[ii] / ct[ii]
                    a0 = np.arctan2(a0[1], a0[0])
                    a1 = vec[ii + 1] * tt[ii] - dvec[ii] / ct[ii]
                    a1 = np.arctan2(a1[1], a1[0])
                    if a1 - a0 > np.pi:
                        a1 -= two_pi
                    elif a1 - a0 < -np.pi:
                        a1 += two_pi
                    n = max(
                        int(np.ceil(abs(a1 - a0) / two_pi * points_per_2pi) + 0.5), 2
                    )
                    a = np.linspace(a0, a1, n)
                    ll = radii[jj][ii] * tt[ii]
                    if ll > 0.49 * length[ii]:
                        r = 0.49 * length[ii] / tt[ii]
                        ll = 0.49 * length[ii]
                    else:
                        r = radii[jj][ii]
                    if ll > 0.49 * length[ii + 1]:
                        r = 0.49 * length[ii + 1] / tt[ii]
                    new_points.extend(
                        r * dvec[ii] / ct[ii]
                        + self.polygons[jj][ii]
                        + np.vstack((r * np.cos(a), r * np.sin(a))).transpose()
                    )
                else:
                    new_points.append(self.polygons[jj][ii])
            self.polygons[jj] = np.array(new_points)
        return self


    def change_layer(self, layer: Optional[int]=None,
                           datatype: Optional[int]=None,
                           color: Optional[str]=None,
                           name: Optional[str] = None) -> Self:
        """
        Change the layer properties of the polygons.
        Only given parameters will be updated in the polygon layer, all the other
        will stay unchanged.

        Args:
            layer : Layer number.
            datatype : Datatype number.
            color : Color hex code.
            name : Name of the layer
        """

        if layer is not None:
            for i in range(len(self.layers)):
                self.layers[i] = layer
        if datatype is not None:
            for i in range(len(self.datatypes)):
                self.datatypes[i] = datatype
        if color is not None:
            for i in range(len(self.colors)):
                self.colors[i] = color
        if name is not None:
            for i in range(len(self.names)):
                self.names[i] = name

        return self
