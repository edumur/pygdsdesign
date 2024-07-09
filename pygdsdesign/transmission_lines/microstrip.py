from functools import partial
import numpy as np
from copy import deepcopy
from typing import Callable, Tuple, Optional, Dict, Union
from typing_extensions import Literal
from scipy.integrate import quad
from scipy.interpolate import interp1d
from tqdm import tqdm

from pygdsdesign.transmission_lines.transmission_line import TransmissionLine
from pygdsdesign.polygons import Rectangle
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.typing_local import Coordinate


class MicroStrip(TransmissionLine):

    def __init__(
        self,
        width: float,
        layer: int = 0,
        datatype: int = 0,
        name: str = "",
        color: str = "",
        ref: Optional[Coordinate] = None,
    ) -> None:
        """
        Microstrip allows to easily draw a continuous microstrip line.

        Parameters
        ----------
        width : float
            Width of the microstrip in um
            This width can be modified latter along the strip or smoothly
            by using tappered functions.
        layer : int
            Layer number of the microstrip.
        datatype : int
            Datatype number of the microstrip.
        """

        TransmissionLine.__init__(self, layer=layer,
                                        datatype=datatype,
                                        name=name,
                                        color=color,
                                        ref=ref)

        self._w = width


    @property
    def width(self):
        return self._w


    @width.setter
    def width(self, width:float):

        self._w = width


    @property
    def param_curve(self):
        return self._param_curve


    ###########################################################################
    #
    #                   Add polygons to the existing microstrip
    #
    ###########################################################################


    def add_line(self, x: float,
                       y: float) -> PolygonSet:
        """
        Add a piece of linear microstrip in the x or y direction .

        Parameters
        ----------
        x : float
            Length of the strip in the x direction in um.
        y : float
            Length of the strip in the y direction in um.
        """

        if x==0:
            if y>0:
                r = Rectangle((self.ref[0], self.ref[1]),
                              (self.ref[0]+self._w, self.ref[1]+y),
                              layer=self._layer,
                              datatype=self._datatype,
                              name=self._name,
                              color=self._color)
            else:
                r = Rectangle((self.ref[0], self.ref[1]),
                                    (self.ref[0]+self._w, self.ref[1]+y),
                                   layer=self._layer,
                                   datatype=self._datatype,
                                   name=self._name,
                                   color=self._color)

            self._add2param([self.ref[0]+self._w/2, self.ref[0]+self._w/2],
                            [self.ref[1], self.ref[1]+y],
                            [0, abs(y)])
        else:
            if x>0:
                r = Rectangle((self.ref[0], self.ref[1]),
                                    (self.ref[0]+x, self.ref[1]+self._w),
                                   layer=self._layer,
                                   datatype=self._datatype,
                                   name=self._name,
                                   color=self._color)
            else:
                r = Rectangle((self.ref[0], self.ref[1]),
                                    (self.ref[0]+x, self.ref[1]+self._w),
                                   layer=self._layer,
                                   datatype=self._datatype,
                                   name=self._name,
                                   color=self._color)

            self._add2param([self.ref[0], self.ref[0]+x],
                            [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                            [0, abs(x)])

        self._add(r)
        self.ref = [self.ref[0]+x, self.ref[1]+y]
        self.total_length += abs(x) + abs(y)

        return self


    ###########################################################################
    #
    #                       Add turn
    #
    ###########################################################################


    def add_turn(self, radius: float,
                       orientation: Literal['lb', 'l-b', 'left-bottom', 'leftbottom',
                                            'lt', 'l-t', 'left-top', 'lefttop'
                                            'rb', 'r-b', 'right-bottom', 'rightbottom'
                                            'rt', 'r-t', 'right-top', 'righttop'
                                            'tl', 't-l', 'top-left', 'topleft'
                                            'tr', 't-r', 'top-right', 'topright'
                                            'bl', 'b-l', 'bottom-left', 'bottomleft'
                                            'br', 'b-r', 'bottom-right', 'bottomright'
                                            'lt', 'l-t', 'lefttop', 'left-top'],
                       nb_points: int=50) -> PolygonSet:
        """
        Add a circulare turn to the strip.

        Parameters
        ----------
        radius : float
            Radius of the arc in um.
        orientation : str
            Orientation of the turn.
            The logic of the orientation naming is the following, the first term
            refers to your departure point while the second one refers to your
            arrival point, they are so the reverse to one another.
            Example: lt _|.
            Also the orientation can be writen in various way, for example all
            following are equivalent:
            'lb', 'l-b', 'left-bottom', 'leftbottom'
        nb_points : int (default=50)
            Number of point used in the polygon.
        """

        if orientation.lower() in ['lb', 'l-b', 'left-bottom', 'leftbottom']:
            theta = np.linspace(np.pi/2., 0, nb_points)
            x0 = self.ref[0]
            y0 = self.ref[1] - radius + self._w/2.

            self.ref = [self.ref[0] + radius - self._w/2., self.ref[1] - radius + self._w/2.]
        elif orientation.lower() in ['lt', 'l-t', 'left-top', 'lefttop']:
            theta = np.linspace(-np.pi/2., 0, nb_points)
            x0 = self.ref[0]
            y0 = self.ref[1] + radius + self._w/2.

            self.ref = [self.ref[0] + radius - self._w/2., self.ref[1] + radius + self._w/2.]
        elif orientation.lower() in ['rb', 'r-b', 'right-bottom', 'rightbottom']:

            theta = np.linspace(np.pi/2., np.pi, nb_points)
            x0 = self.ref[0]
            y0 = self.ref[1] - radius + self._w/2.

            self.ref = [self.ref[0] - radius - self._w/2., self.ref[1] - radius + self._w/2.]
        elif orientation.lower() in ['rt', 'r-t', 'right-top', 'righttop']:

            theta = np.linspace(-np.pi/2., -np.pi, nb_points)
            x0 = self.ref[0]
            y0 = self.ref[1] + radius + self._w/2.

            self.ref = [self.ref[0] - radius - self._w/2., self.ref[1] + radius + self._w/2.]
        elif orientation.lower() in ['tl', 't-l', 'top-left', 'topleft']:

            theta = np.linspace(0, -np.pi/2., nb_points)
            x0 = self.ref[0] - radius + self._w/2.
            y0 = self.ref[1]

            self.ref = [self.ref[0] - radius + self._w/2., self.ref[1] - radius - self._w/2.]
        elif orientation.lower() in ['tr', 't-r', 'top-right', 'topright']:

            theta = np.linspace(-np.pi, -np.pi/2., nb_points)
            x0 = self.ref[0] + radius + self._w/2.
            y0 = self.ref[1]

            self.ref = [self.ref[0] + radius + self._w/2., self.ref[1] - radius - self._w/2.]
        elif orientation.lower() in ['bl', 'b-l', 'bottom-left', 'bottomleft']:
            theta = np.linspace(0, np.pi/2., nb_points)
            x0 = self.ref[0] - radius + self._w/2.
            y0 = self.ref[1]

            self.ref = [self.ref[0] - radius + self._w/2., self.ref[1] + radius - self._w/2.]
        elif orientation.lower() in ['br', 'b-r', 'bottom-right', 'bottomright']:
            theta = np.linspace(np.pi, np.pi/2., nb_points)
            x0 = self.ref[0] + radius + self._w/2.
            y0 = self.ref[1]

            self.ref = [self.ref[0] + radius + self._w/2., self.ref[1] + radius - self._w/2.]
        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.')

        x = np.concatenate(((radius-self._w/2.)*np.cos(theta), (radius + self._w/2.)*np.cos(theta[::-1])))
        y = np.concatenate(((radius-self._w/2.)*np.sin(theta), (radius + self._w/2.)*np.sin(theta[::-1])))

        p = np.vstack((x0+x, y0+y)).T

        self._add(PolygonSet(polygons=[p],
                             layers=[self._layer],
                             datatypes=[self._datatype],
                             names=[self._name],
                             colors=[self._color]))

        self.total_length += radius*np.pi/2.

        self._add2param(x0+radius*np.cos(theta),
                        y0+radius*np.sin(theta),
                        np.linspace(0, 1, nb_points)*radius*np.pi/2)

        return self


    def add_fresnel_turn(self, radius: float,
                               orientation: Literal['lb', 'l-b', 'left-bottom', 'leftbottom',
                                                    'lt', 'l-t', 'left-top', 'lefttop'
                                                    'rb', 'r-b', 'right-bottom', 'rightbottom'
                                                    'rt', 'r-t', 'right-top', 'righttop'
                                                    'tl', 't-l', 'top-left', 'topleft'
                                                    'tr', 't-r', 'top-right', 'topright'
                                                    'bl', 'b-l', 'bottom-left', 'bottomleft'
                                                    'br', 'b-r', 'bottom-right', 'bottomright'
                                                    'lt', 'l-t', 'lefttop', 'left-top'],
                               nb_points: int=51) -> PolygonSet:
        """
        Return the parametric Fresnel curve along the length t.
        The curve is doubled, mirrored, joint and normalized to a given radius

        Parameters
        ----------
        radius : float
            Radius of the arc in um.
        orientation : str
            Orientation of the turn.
            The logic of the orientation naming is the following, the first term
            refers to your departure point while the second one refers to your
            arrival point, they are so the reverse to one another.
            Example: lt _|.
            Also the orientation can be writen in various way, for example all
            following are equivalent:
            'lb', 'l-b', 'left-bottom', 'leftbottom'
        nb_points : int (default=51)
            Number of point used in the polygon.
            Must be an odd number
        """



        # All the computations are stored in self.calc to cache the result
        dx, dy, t, x, y, f_dx, f_dy, length = self.calc(radius, nb_points)

        # Prepare functions for the parametric generation
        # Initialize func and dfunc with some values to reuse them later on
        partial_func = partial(self.func, x=x,y=y)
        partial_dfunc = partial(self.dfunc, dx=dx,dy=dy,f_dx=f_dx,f_dy=f_dy)

        # Get the polygon corresponding to a Fresnel turn
        # Add_length is set to False, because of its computational time.
        # We do the computation here and save it for later
        p = self.add_parametric_curve(partial_func, partial_dfunc, t,
                args=None,
                add_polygon=False,
                add_length=False)


        # Add the length of the parametric curve only if asked (default)
        if hasattr(self, '_fresnel_lengths'):
            if radius not in self._fresnel_lengths.keys():
                compute_length = True
            else:
                compute_length = False
        else:
            self._fresnel_lengths: Dict[float, float]= {}
            compute_length = True

        if compute_length:
            self._fresnel_lengths[radius] = length

        self.total_length += self._fresnel_lengths[radius]
        t = np.linspace(0, 1, nb_points)*self._fresnel_lengths[radius]

        # We transform the polygon and move the ref depending on the orientation
        if orientation.lower() in ['lt', 'l-t', 'lefttop', 'left-top']:
            self._add2param(self.ref[0]+x[::-1], self.ref[1]+y[::-1], t)
            self.ref = p.polygons[0][int(nb_points-1)]
            self._add(p)
        elif orientation.lower() in ['lb', 'l-b', 'leftbottom', 'left-bottom']:
            p.flip('x')
            p.translate(0, -radius+self._w/2.)
            self._add2param(self.ref[0]+x[::-1], self.ref[1]-y[::-1]+self._w, t)
            self.ref = [p.polygons[0][nb_points][0]-self._w, p.polygons[0][nb_points][1]]
            self._add(p)
        elif orientation.lower() in ['br', 'b-r', 'bottomright', 'bottom-right']:
            p.flip('x')
            p.flip('y')
            self._add2param(self.ref[0]+y[::-1], self.ref[1]+x[::-1], t)
            self.ref = p.polygons[0][0]
            self._add(p)
        elif orientation.lower() in ['bl', 'b-l', 'bottomright', 'bottom-right']:
            p.rotate(np.pi/2., p.polygons[0][0]+self._w/2)
            p.translate(-self._w, -self._w)
            self._add2param(self.ref[0]-y[::-1]+self._w, self.ref[1]+x[::-1], t)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]-self._w]
            self._add(p)
        elif orientation.lower() in ['tr', 't-r', 'topright', 'top-right']:
            p.rotate(-np.pi/2., p.polygons[0][0]+self._w/2)
            p.translate(self._w, -2*self._w)
            self._add2param(self.ref[0]+y[::-1], self.ref[1]-x[::-1], t)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]]
            self._add(p)
        elif orientation.lower() in ['tl', 't-l', 'topleft', 'top-left']:
            p.rotate(-np.pi/2., p.polygons[0][0]+self._w/2)
            p.flip('y')
            p.translate(-radius+3*self._w/2.,-2*self._w)
            self._add2param(self.ref[0]-y[::-1]+self._w, self.ref[1]-x[::-1], t)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]]
            self._add(p)
        elif orientation.lower() in ['rb', 'r-b', 'rightbottom', 'right-bottom']:
            p.flip('x').flip('y')
            p.translate(-radius-self._w/2, -radius+self._w/2)
            self._add2param(self.ref[0]-x[::-1], self.ref[1]-y[::-1]+self._w, t)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]]
            self._add(p)
        elif orientation.lower() in ['rt', 'r-t', 'righttop', 'right-top']:
            p.flip('y')
            p.translate(-radius-self._w/2, 0)
            self._add2param(self.ref[0]-x[::-1], self.ref[1]+y[::-1], t)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]]
            self._add(p)
        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.')
        return self


    ###########################################################################
    #
    #                               Tapers
    #
    ###########################################################################


    def add_taper(self, x: float,
                        y: float,
                        new_width: float) -> PolygonSet:
        """
        Add linear taper between the current and the new width.

        Parameters
        ----------
        x : float
            Length of the taper in the x direction in um.
        y : float
            Length of the taper in the y direction in um.
        new_width : float
            New width of the microstrip in um.
        """

        if x == 0 and y > 0:
            p = [(self.ref[0], self.ref[1]),
                 (self.ref[0]-new_width/2.+self._w/2., self.ref[1]+y),
                 (self.ref[0]+new_width/2.+self._w/2., self.ref[1]+y),
                 (self.ref[0]+self._w, self.ref[1])]

            self._add2param([self.ref[0]+self._w/2, self.ref[0]+self._w/2],
                            [self.ref[1], self.ref[1]+y],
                            [0, abs(y)])
            self.ref = [self.ref[0]-new_width/2.+self._w/2., self.ref[1]+y]
        elif x == 0 and y < 0:
            p = [(self.ref[0], self.ref[1]),
                 (self.ref[0]-new_width/2. +self._w/2., self.ref[1]+y),
                 (self.ref[0]+new_width/2. +self._w/2., self.ref[1]+y),
                 (self.ref[0]+self._w, self.ref[1])]

            self._add2param([self.ref[0]+self._w/2, self.ref[0]+self._w/2],
                            [self.ref[1], self.ref[1]+y],
                            [0, abs(y)])
            self.ref = [self.ref[0]-new_width/2.+self._w/2., self.ref[1]+y]
        elif x > 0 and y==0:

            p = [(self.ref[0], self.ref[1]),
                 (self.ref[0]+x, self.ref[1] - new_width/2.+self._w/2.),
                 (self.ref[0]+x, self.ref[1] + new_width/2.+self._w/2.),
                 (self.ref[0], self.ref[1]+self._w)]

            self._add2param([self.ref[0], self.ref[0]+x],
                            [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                            [0, abs(x)])
            self.ref = [self.ref[0]+x, self.ref[1]-new_width/2.+self._w/2.]
        elif x<0 and y==0:

            p = [(self.ref[0], self.ref[1]),
                 (self.ref[0]+x, self.ref[1] - new_width/2.+self._w/2.),
                 (self.ref[0]+x, self.ref[1] + new_width/2.+self._w/2.),
                 (self.ref[0], self.ref[1]+self._w)]

            self._add2param([self.ref[0], self.ref[0]+x],
                            [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                            [0, abs(x)])
            self.ref = [self.ref[0]+x, self.ref[1]-new_width/2.+self._w/2.]

        self._add(PolygonSet(polygons=[p],
                             layers=[self._layer],
                             datatypes=[self._datatype],
                             names=[self._name],
                             colors=[self._color]))

        self.total_length += abs(x) + abs(y)
        self._w = new_width

        return self


    def add_taper_cosec(self, x: float,
                              y: float,
                              new_width: float,
                              smoothness: float=1,
                              nb_points: int=50) -> PolygonSet:
        """
        Add smooth taper between the current and the new width.
        The equation of the smooth curve is based on the cosec function

        Parameters
        ----------
        x : float
            Length of the taper in the x direction in um.
        y : float
            Length of the taper in the y direction in um.
        new_width : float
            New width of the microstrip in um.
        smoothness : float (default 1)
            Slop of the taper, smaller number implying sharper transition.
        nb_points : int (default=50)
            Number of point used in the polygon.
        """

        # Calculate curve ensuring it goes from y=0 to y=1
        tx  = np.linspace(0., abs(x)+abs(y), nb_points)
        ty = 1./((1.+np.exp((tx-(abs(x)+abs(y))/2.)/smoothness))/np.exp((tx-(abs(x)+abs(y))/2.)/smoothness))
        ty -= ty.min()
        ty /= ty.max()

        # Scale curve to proper microstrip width and tapering
        ty = self._w/2. + (new_width-self._w)*ty/2.

        # Build ref polygon
        tx = np.concatenate((tx, tx[::-1]))
        ty = np.concatenate((ty, ty[::-1]*-1))
        p = PolygonSet(polygons=[np.vstack((tx, ty)).T],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])

        # Transform polygon depending on orientation
        if y==0 and x>0:
            p.translate(self.ref[0], self.ref[1]+self._w/2.)
            self._add2param([self.ref[0], self.ref[0]+x],
                            [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                            [0, abs(x)])
            self.ref = [self.ref[0]+x, self.ref[1]-new_width/2.+self._w/2.]
        elif y==0 and x<0:
            p.polygons[0]=np.vstack((p.polygons[0][:,0]*-1, p.polygons[0][:,1])).T
            p.translate(self.ref[0], self.ref[1]+self._w/2.)
            self._add2param([self.ref[0], self.ref[0]+x],
                            [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                            [0, abs(x)])
            self.ref = [self.ref[0]+x, self.ref[1]-new_width/2.+self._w/2.]
        elif x==0 and y>0:
            p.translate(0, -self._w/2.)
            p.rotate(np.pi/2., center=(0, 0))
            p.translate(self.ref[0], self.ref[1])
            self._add2param([self.ref[0]+self._w/2, self.ref[0]+self._w/2],
                            [self.ref[1], self.ref[1]+y],
                            [0, abs(y)])
            self.ref = [self.ref[0]-new_width/2.+self._w/2., self.ref[1]+y]
        elif x==0 and y<0:
            p.rotate(-np.pi/2., center=(0, 0))
            p.translate(self._w/2., 0)
            p.translate(self.ref[0], self.ref[1])
            self._add2param([self.ref[0]+self._w/2, self.ref[0]+self._w/2],
                            [self.ref[1], self.ref[1]+y],
                            [0, abs(y)])
            self.ref = [self.ref[0]-new_width/2.+self._w/2., self.ref[1]+y]

        self._add(p)

        self.total_length += abs(x) + abs(y)
        self._w = new_width

        return self


    ###########################################################################
    #
    #                   Generic parametric curve
    #
    ###########################################################################


    def add_parametric_curve(self, f: Callable[..., Tuple[np.ndarray, np.ndarray]],
                                  df: Callable[..., Tuple[np.ndarray, np.ndarray]],
                                   t: np.ndarray,
                                args: Optional[Tuple[Optional[float], ...]]=None,
                                add_polygon: bool=True,
                                add_length: bool=True) -> PolygonSet:
        """
        Create a microstrip line following the parametric equation f and its
        derivative df along the length t.
        In order to return the curve length correctly, the derivative df of f
        must be correct, its absolute amplitude must be correct.

        Parameters
        ---------
        f : func
            Function calculating the parametric equation.
            Must be of the type f(t, args) and return a tuple of coordinate
            (x, y).
        df : func
            Function calculating the derivative of the parametric equation.
            Must be of the type df(t, args) and return a tuple of coordinate
            (dx, dy).
        t : np.ndarray
            Array of the length of the parametric curve.
            Also determine the number of point of the total polygon.
            Must not necessarily starts at 0.
        args : variable arguments (default None)
            Argument passed to f and df
        """
        if args is None:
            args = (None, )

        dx1, dy1 = df(t, args)

        n = np.hypot(dx1, dy1)

        dx1, dy1 = dx1/n, dy1/n

        x1_, y1_ = f(t, args)
        theta1 = np.angle(dx1+1j*dy1)-np.pi/2.
        x1, y1 = x1_+np.cos(theta1)*self._w/2., y1_+np.sin(theta1)*self._w/2.


        dx2, dy2 = df(t[::-1], args)
        n = np.hypot(dx2, dy2)
        dx2, dy2 = dx2/n, dy2/n
        x2_, y2_ = f(t[::-1], args)
        theta2 = np.angle(dx2+1j*dy2)+np.pi/2.
        x2, y2 = x2_+np.cos(theta2)*self._w/2., y2_+np.sin(theta2)*self._w/2.

        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
        p = np.vstack((x, y)).T

        poly = PolygonSet(polygons=[p],
                         layers=[self._layer],
                         datatypes=[self._datatype],
                         names=[self._name],
                         colors=[self._color]).translate(self.ref[0], self.ref[1])

        # Calculate curve length
        def func(t, args):
            dx1, dy1 = df(t, args)
            return np.hypot(dx1, dy1)

        # Add the length of the parametric curve only if asked (default)
        if add_length:
            self.total_length += quad(func, t[0], t[-1], args=(args,))[0]

        # Add polygon only if asked (default)
        if add_polygon:
            self._add2param(self.ref[0]+x1_,
                            self.ref[1]+y1_,
                            np.linspace(0, 1, len(t))*quad(func, t[0], t[-1], args=(args,))[0])
            self.ref = poly.polygons[0][int(len(poly.polygons[0])/2)]
            self._add(poly)

            return self
        else:
            return poly


    ###########################################################################
    #
    #                   Added structures following the strip
    #
    ###########################################################################


    def add_and_follow(self, structure: Union[PolygonSet, PolygonSet],
                             pitch: float=100.,
                             start_offset: float=0.,
                             stop_offset: float=0.,
                             distance: float=50.,
                             angle: float=0.,
                             noise: float=0.,
                             nb_point_linearization: int=1000,
                             add_polygon: bool=True,
                             verbose: bool=False) -> PolygonSet:
        """
        Add structures along the line. The structures follow the curve and are
        rotating accordingly.

        Args:
            structure : Structure to be added along the path.
            pitch: pitch between two adjacent structures in um.
                Defaults to 100um.
            start_offset: Offset between the beginning of the microstrip and the
                first added structure in um.
                Defaults to 0um.
            stop_offset: Offset between the end of the microstrip and the last
                added structure in um.
                Defaults to 0um.
            distance: distance between the center of the microstrip and the
                center of the added structure in um. This distance is refered to
                the orthogonal vector of the line. A positive distance shift
                structures on the "right" of the microstrip.
                Defaults to 50um.
            angle: angle used to rotate the structure in radian
                Defaults to 0rad.
            noise: Nornal noise added to the pitch of the added structures in um.
                Usefull to avoid a perfect distance between the added structures
                and spurious resonant effect.
                Defaults to 0um.
            nb_point_linearization: Internally the method linearizes the
                parametric curve describing the shape of the microstrip. This
                linearization requires a large number of points in order to
                obtain a precise estimation of the parametric curve.
                Defaults to 1000.
            add_polygon: If True add the added structures to the microstrip.
                If False, return the added structures as independent polygons.
                Defaults to True.
            verbose: If true, display a progress bar in the terminal. Usefull
                for time consuming rendering.
                Deafault to False.
        """

        # Center the structure
        structure.center()

        # First you need to linearize the parametric coordinate
        t = np.linspace(start_offset, self.param_curve[:,2][-1], nb_point_linearization)
        x = np.interp(t, self.param_curve[:,2], self.param_curve[:,0])
        y = np.interp(t, self.param_curve[:,2], self.param_curve[:,1])


        # From the linearized coordinates, we create coordinate functions
        f_x = interp1d(t, x)
        f_y = interp1d(t, y)

        f_dx = interp1d(t, np.gradient(x, t))
        f_dy = interp1d(t, np.gradient(y, t))

        f_theta = lambda t: np.angle(f_dx(t)+1j*f_dy(t))

        # For every pitch along the parametric curve, we add the given structure
        ts = np.linspace(start_offset,
                         self.param_curve[:,2][-1]-stop_offset,
                         int((self.param_curve[:,2][-1]-stop_offset)/pitch))

        # We add normal noise to the computed position
        # By default, noise is 0, adding no noise
        # If noise is not 0, we take care of out of range value with masks
        ts += np.random.normal(0, noise, len(ts))
        ts[ts<start_offset] = start_offset
        ts[ts>self.param_curve[:,2][-1]-stop_offset] = self.param_curve[:,2][-1]-stop_offset

        if verbose:
            it = tqdm
        else:
            it = lambda t: t

        poly = PolygonSet()
        for t in it(ts):

            # Compute the coordinate and angle of the curve at the distance t
            x1, y1, theta = f_x(t), f_y(t), f_theta(t)

            # With the angle, we compute the added coordinates giving the
            # wanted distance
            # The added -pi/2 is to give the distance in reference to the orthogonal
            # direction in respect to the parametric curve.
            x1 += np.cos(theta-np.pi/2)*distance
            y1 += np.sin(theta-np.pi/2)*distance

            # Add the structure at (order matters):
            # 1. the proper angle
            # 2. the proper coordinate
            poly += deepcopy(structure)\
                    .rotate(theta+angle, (0, 0))\
                    .translate(x1, y1)

        # Add polygon only if asked (default)
        if add_polygon:
            self += poly
            return self
        else:
            return poly
