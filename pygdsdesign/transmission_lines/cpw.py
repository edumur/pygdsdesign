from functools import partial
import numpy as np
import copy
from typing import Callable, Tuple, Optional, Dict, Union
from typing_extensions import Literal
from scipy.integrate import quad

from pygdsdesign.polygons import Rectangle
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.transmission_lines.transmission_line import TransmissionLine
from pygdsdesign.typing_local import Coordinate


class CPW(TransmissionLine):

    def __init__(
        self,
        width: float,
        gap: float,
        layer: int = 0,
        datatype: int = 0,
        name: str = "",
        color: str = "",
        ref: Optional[Coordinate] = None,
    ) -> None:
        """
        Coplanar allows to easily draw a continuous coplanar waveguide.

        Parameters
        ----------
        width : float
            Width of the central line in um
            This width can be modified latter along the strip or smoothly
            by using tappered functions.
        gap : float
            Width of the central line in um
            This width can be modified latter along the strip or smoothly
            by using tappered functions.
        layer : int
            Layer number of the coplanar.
        datatype : int
            Datatype number of the coplanar.
        """

        TransmissionLine.__init__(self, layer=layer,
                                        datatype=datatype,
                                        name=name,
                                        color=color,
                                        ref=ref)

        self._w = width
        self._s = gap
        self._bounding_polygon = PolygonSet()


    @property
    def width(self):
        return self._w


    @width.setter
    def width(self, width:float):
        self._w = width


    @property
    def gap(self):
        return self._s


    @gap.setter
    def gap(self, gap:float):
        self._s = gap


    @property
    def bounding_polygon(self):
        return self._bounding_polygon


    # def translate(self, dx: float,
    #                     dy: float) -> PolygonSet:
    #     """
    #     Translate the cpw by the amount dx, dy in the x and y direction.
    #     Take care of translating also the bounding_polygon.

    #     Args:
    #         dx (float): amount of translation in the x direction in um.
    #         dy (float): amount of translation in the y direction in um.
    #     """

    #     self._bounding_polygon.translate(dx,dy)
    #     return super().translate(dx, dy)


    ###########################################################################
    #
    #                   Add polygons to the existing coplanar waveguide
    #
    ###########################################################################


    def add_line(self, x: float,
                       y: float) -> PolygonSet:
        """
        Add a piece of linear coplanar in the x or y direction .

        Parameters
        ----------
        x : float
            Length of the strip in the x direction in um.
        y : float
            Length of the strip in the y direction in um.
        """
        p  = PolygonSet([[(0, -self._w/2.),
                          (0, -self._w/2. - self._s),
                          (x+y, -self._w/2. - self._s),
                          (x+y, -self._w/2.)]],
                         layers=[self._layer],
                         datatypes=[self._datatype],
                         names=[self._name],
                         colors=[self._color])

        p += PolygonSet([[(0, +self._w/2.),
                            (0, +self._w/2. + self._s),
                            (x+y, +self._w/2. + self._s),
                            (x+y, +self._w/2.)]],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color])

        if y==0 and x>0:
            pass
        elif y==0 and x<0:
            pass
        elif y>0 and x==0:
            p.rotate(np.pi/2.)
        elif y<0 and x==0:
            p.rotate(np.pi/2.)
        else:
            raise ValueError("x and y can't be both != 0")

        self._add(p.translate(*self.ref))
        self.ref = [self.ref[0]+x, self.ref[1]+y]
        self.total_length += abs(x) + abs(y)

        # update bounding polygon
        a,b = p.get_bounding_box()
        bp = Rectangle((a[0], a[1]),
                        (b[0], b[1]))
        self._bounding_polygon+=bp

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
        nb_point : int (default=50)
            Number of point used in the polygon.
        """


        theta = np.linspace(np.pi/2., 0, nb_points)

        x = np.concatenate(((radius+self._w/2.)*np.cos(theta), (radius+self._w/2.+self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius+self._w/2.)*np.sin(theta), (radius+self._w/2.+self._s)*np.sin(theta[::-1])))

        p = PolygonSet(polygons=[np.vstack((x, y)).T],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])

        x = np.concatenate(((radius-self._w/2.)*np.cos(theta), (radius-self._w/2.-self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius-self._w/2.)*np.sin(theta), (radius-self._w/2.-self._s)*np.sin(theta[::-1])))

        p += PolygonSet(polygons=[np.vstack((x, y)).T],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color])

        # generate bounding polygon
        x = np.concatenate(((radius+self._w/2.+self._s)*np.cos(theta), (radius-self._w/2.-self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius+self._w/2.+self._s)*np.sin(theta), (radius-self._w/2.-self._s)*np.sin(theta[::-1])))

        bp = PolygonSet(polygons=[np.vstack((x, y)).T])

        if orientation.lower() in ['lb', 'l-b', 'left-bottom', 'leftbottom']:
            p.translate(self.ref[0], self.ref[1]-radius)
            bp.translate(self.ref[0], self.ref[1]-radius)
            self.ref = [self.ref[0] + radius, self.ref[1] - radius]

        elif orientation.lower() in ['lt', 'l-t', 'left-top', 'lefttop']:
            p.rotate(-np.pi/2.).translate(self.ref[0], self.ref[1]+radius)
            bp.rotate(-np.pi/2.).translate(self.ref[0], self.ref[1]+radius)
            self.ref = [self.ref[0] + radius, self.ref[1] + radius]

        elif orientation.lower() in ['rb', 'r-b', 'right-bottom', 'rightbottom']:
            p.rotate(-3.*np.pi/2.).translate(self.ref[0], self.ref[1]-radius)
            bp.rotate(-3.*np.pi/2.).translate(self.ref[0], self.ref[1]-radius)
            self.ref = [self.ref[0] - radius, self.ref[1] - radius]

        elif orientation.lower() in ['rt', 'r-t', 'right-top', 'righttop']:
            p.rotate(np.pi).translate(self.ref[0], self.ref[1]+radius)
            bp.rotate(np.pi).translate(self.ref[0], self.ref[1]+radius)
            self.ref = [self.ref[0] - radius, self.ref[1] + radius]

        elif orientation.lower() in ['tl', 't-l', 'top-left', 'topleft']:
            p.translate(-radius+self.ref[0], -radius-self._s-self._w/2.+self.ref[1]).flip('x')
            bp.translate(-radius+self.ref[0], -radius-self._s-self._w/2.+self.ref[1]).flip('x')
            self.ref = [self.ref[0] - radius, self.ref[1] - radius]

        elif orientation.lower() in ['tr', 't-r', 'top-right', 'topright']:
            p.rotate(np.pi).translate(self.ref[0]+radius, self.ref[1])
            bp.rotate(np.pi).translate(self.ref[0]+radius, self.ref[1])
            self.ref = [self.ref[0] + radius, self.ref[1] - radius]

        elif orientation.lower() in ['bl', 'b-l', 'bottom-left', 'bottomleft']:
            p.translate(self.ref[0]-radius, self.ref[1])
            bp.translate(self.ref[0]-radius, self.ref[1])
            self.ref = [self.ref[0] - radius, self.ref[1] + radius]

        elif orientation.lower() in ['br', 'b-r', 'bottom-right', 'bottomright']:
            p.rotate(np.pi/2.).translate(self.ref[0]+radius, self.ref[1])
            bp.rotate(np.pi/2.).translate(self.ref[0]+radius, self.ref[1])
            self.ref = [self.ref[0] + radius, self.ref[1] + radius]
        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.')

        self._add(p)
        self._bounding_polygon+=bp
        self.total_length += radius*np.pi/2.

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
                               nb_points: int=101) -> PolygonSet:
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
        nb_points : int (default=101)
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
        p, bp = self.add_parametric_curve(partial_func,
                                          partial_dfunc,
                                          t,
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

        # We transform the polygon and move the ref depending on the orientation
        if orientation.lower() in ['lt', 'l-t', 'lefttop', 'left-top']:
            p.translate(0, -self._w/2.)
            bp.translate(0, -self._w/2.)
            self.ref = [p.polygons[0][nb_points][0]+self._w/2., p.polygons[0][nb_points][1]]

        elif orientation.lower() in ['lb', 'l-b', 'leftbottom', 'left-bottom']:
            p.flip('x')
            p.translate(0, -radius+self._s)
            bp.flip('x')
            bp.translate(0, -radius+self._s)
            self.ref = [p.polygons[0][nb_points][0]+self._w/2., p.polygons[0][nb_points][1]]

        elif orientation.lower() in ['br', 'b-r', 'bottomright', 'bottom-right']:
            p.flip('x')
            p.flip('y')
            p.translate(-self._s-self._w/2., self._s)
            bp.flip('x')
            bp.flip('y')
            bp.translate(-self._s-self._w/2., self._s)
            self.ref = [p.polygons[0][-1][0], p.polygons[0][-1][1]+self._w/2.]

        elif orientation.lower() in ['bl', 'b-l', 'bottomright', 'bottom-right']:
            p.rotate(np.pi/2., p.polygons[1][0])
            p.translate(self._s+self._w/2., self._s)
            bp.rotate(np.pi/2., bp.polygons[0][0])
            bp.translate(-(self._s+self._w/2), -(self._s+self._w))
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]+self._w/2.]

        elif orientation.lower() in ['tr', 't-r', 'topright', 'top-right']:
            p.rotate(-np.pi/2., p.polygons[1][0])
            p.translate(-self._s-self._w/2., self._s)
            bp.rotate(-np.pi/2., bp.polygons[0][0])
            bp.translate(self._s+self._w/2., -(self._w + self._s))
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]-self._w/2.]

        elif orientation.lower() in ['tl', 't-l', 'topleft', 'top-left']:
            p.rotate(-np.pi/2., p.polygons[1][0])
            p.flip('y')
            p.translate(-radius, self._s)
            bp.rotate(-np.pi/2., bp.polygons[0][0])
            bp.flip('y')
            bp.translate(-radius+self._w+2*self._s, -self._w - self._s)
            self.ref = [p.polygons[0][nb_points][0], p.polygons[0][nb_points][1]-self._w/2.]

        elif orientation.lower() in ['rb', 'r-b', 'rightbottom', 'right-bottom']:
            p.rotate(np.pi, p.polygons[1][0])
            p.translate(0,self._s*2.+self._w/2.)
            bp.rotate(np.pi, bp.polygons[0][0])
            bp.translate(0,-(self._s*2.+1.5*self._w))
            self.ref = [p.polygons[0][nb_points][0]-self._w/2., p.polygons[0][nb_points][1]]

        elif orientation.lower() in ['rt', 'r-t', 'righttop', 'right-top']:
            p.flip('y')
            p.translate(-radius-self._s-self._w/2., -self._w/2.)
            bp.flip('y')
            bp.translate(-radius-self._s-self._w/2., -self._w/2.)
            self.ref = [p.polygons[0][nb_points][0]-self._w/2., p.polygons[0][nb_points][1]]

        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.')

        self._add(p)
        self._bounding_polygon+=bp
        return self


    ###########################################################################
    #
    #                               Tapers
    #
    ###########################################################################


    def add_taper(self, x: float,
                        y: float,
                        new_width: float,
                        new_gap: float) -> PolygonSet:
        """
        Add linear taper between the current and the new width.

        Parameters
        ----------
        x : float
            Length of the taper in the x direction in um.
        y : float
            Length of the taper in the y direction in um.
        new_width : float
            New width of the coplanar waveguide in um.
        new_gape : float
            New gape of the coplanar waveguide in um.
        """

        p = PolygonSet(polygons=[[(0., self._w/2.)]],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])
        p>(x+y, new_width/2.-self._w/2.)
        p>(0., new_gap)
        p>(-x-y, -new_gap-new_width/2.+self._s+self._w/2.)
        p += copy.copy(p).mirror((0, 0), (1, 0))

        # generate bounding polygon
        bp = PolygonSet(polygons=[[(0, self._w/2+self._s),
                                  (x+y, new_width/2+new_gap),
                                  (x+y, 0),
                                  (0, 0)]]
                          )
        bp += copy.copy(bp).mirror((0, 0), (1, 0))

        if x>0 and y==0:
            p.translate(*self.ref)
            bp.translate(*self.ref)
        elif x<0 and y==0:
            p.translate(*self.ref)
            bp.translate(*self.ref)
        elif x==0 and y<0:
            p.rotate(np.pi/2.).translate(*self.ref)
            bp.rotate(np.pi/2.).translate(*self.ref)
        elif x==0 and y>0:
            p.rotate(np.pi/2.).translate(*self.ref)
            bp.rotate(np.pi/2.).translate(*self.ref)
        else:
            raise ValueError('x and y should not be both != than 0.')


        self.ref = [self.ref[0]+x, self.ref[1]+y]

        self._add(p)
        self._bounding_polygon+=bp

        self.total_length += abs(x) + abs(y)
        self._w = new_width
        self._s = new_gap

        return self


    def add_parametric_curve(self,
                             f: Callable[..., Tuple[np.ndarray, np.ndarray]],
                             df: Callable[..., Tuple[np.ndarray, np.ndarray]],
                             t: np.ndarray,
                             args: Optional[Tuple[Optional[float], ...]]=None,
                             add_polygon: bool=True,
                             add_length: bool=True) -> Union[PolygonSet, Tuple[PolygonSet, PolygonSet]]:
        """
        Create a coplanar line following the parametric equation f and its
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

        x1, y1 = f(t, args)
        theta1 = np.angle(dx1+1j*dy1)-np.pi/2.
        x1, y1 = x1+np.cos(theta1)*(self._w/2.+self._s), y1+np.sin(theta1)*(self._w/2.+self._s)

        dx2, dy2 = df(t[::-1], args)
        n = np.hypot(dx2, dy2)
        dx2, dy2 = dx2/n, dy2/n
        x2, y2 = f(t[::-1], args)
        theta2 = np.angle(dx2+1j*dy2)-np.pi/2.
        x2, y2 = x2+np.cos(theta2)*self._w/2., y2+np.sin(theta2)*self._w/2.

        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
        p1 = np.vstack((x, y)).T
        # keep the coordinates of the outer trace
        bp_x, bp_y = x1, y1

        dx1, dy1 = df(t, args)
        n = np.hypot(dx1, dy1)
        dx1, dy1 = dx1/n, dy1/n

        x1, y1 = f(t, args)
        theta1 = np.angle(dx1+1j*dy1)-np.pi/2.
        x1, y1 = x1+np.cos(theta1)*-(self._w/2.+self._s), y1+np.sin(theta1)*-(self._w/2.+self._s)


        dx2, dy2 = df(t[::-1], args)
        n = np.hypot(dx2, dy2)
        dx2, dy2 = dx2/n, dy2/n
        x2, y2 = f(t[::-1], args)
        theta2 = np.angle(dx2+1j*dy2)-np.pi/2.
        x2, y2 = x2+np.cos(theta2)*-self._w/2., y2+np.sin(theta2)*-self._w/2.

        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
        p2 = np.vstack((x, y)).T
        # add the coordinates of the outer trace in reversed order
        bp_x, bp_y = np.concatenate((bp_x, np.flip(x1))), np.concatenate((bp_y, np.flip(y1)))
        bp = np.vstack((bp_x, bp_y)).T

        p = PolygonSet(polygons=[p1],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])\
           + PolygonSet(polygons=[p2],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color])
        p = p.translate(self.ref[0], self.ref[1])

        # generate bounding_polygon
        bp = PolygonSet(polygons=[bp])
        bp = bp.translate(self.ref[0], self.ref[1])

        # Calculate curve length
        def func(t, args):
            dx1, dy1 = df(t, args)
            return np.hypot(dx1, dy1)

        # Add the length of the parametric curve only if asked (default)
        if add_length:
            self.total_length += quad(func, t[0], t[-1], args=(args,))[0]

        # Add polygon only if asked (default)
        if add_polygon:
            self.ref = [p.polygons[0][int(len(p.polygons[0])/2)][0]-self._w/2.,
                        p.polygons[0][int(len(p.polygons[0])/2)][1]]
            self._add(p)
            self._bounding_polygon+=bp

            return self
        else:
            return p, bp


    def add_end(self, x: float,
                      y: float,
                      update_ref: bool=False) -> PolygonSet:
        """
        Add an open end to a coplanar waveguide in the x, y direction
        Notice that this method does not increment the total length of the
        waveguide nor its reference point (unless update_ref is set to True)!

        Parameters
        ----------
        x : float
            Length of the end in the x direction in um.
        y : float
            Length of the end in the y direction in um.
        update_ref : bool
            If True update the reference point of the cpw by including the length
            of the end in x and y direction.
            Useful when starting a cpw by an end_point.
        """

        if x ==0.:
            r = Rectangle((-self._w/2.-self._s, 0),
                          (self._w/2.+self._s, y),
                          layer=self._layer,
                          datatype=self._datatype,
                          name=self._name,
                          color=self._color)
        elif y==0.:
            r = Rectangle((0, -self._w/2.-self._s),
                          (x,  self._w/2.+self._s),
                          layer=self._layer,
                          datatype=self._datatype,
                          name=self._name,
                          color=self._color)
        else:
            raise ValueError('x and y cannot be both different than 0.')

        self._add(r.translate(*self.ref))

        if update_ref:
            self.ref = [self.ref[0]+x, self.ref[1]+y]

        # update bounding polygon
        a,b = r.get_bounding_box()
        bp = Rectangle((a[0], a[1]),
                        (b[0], b[1]))
        self._bounding_polygon+=bp

        return self


    def add_circular_end(self, orientation:str,
                               update_ref: bool=False,
                               nb_points: int=50) -> PolygonSet:
        """
        Add a circular open end to a coplanar waveguide in the given
        orientation.
        Notice that this method does not increment the total length of the
        waveguide nor its reference point (unless update_ref is set to True)!

        Parameters
        ----------
        orientation : str {'left', right', 'top', 'bottom'}
            Orientation of the open end.
            Also the orientation can be writen in various way, for instance
            following are equivalent:
            'l', 'left'
        update_ref : bool
            If True update the reference point of the cpw by including the length
            of the end in x and y direction.
            Useful when starting a cpw by an end_point.
        nb_point : int (default=50)
            Number of point used in the polygon.
        """

        theta = np.linspace(-np.pi/2, np.pi/2, nb_points)
        x = np.concatenate(((self._w/2.)*np.cos(theta), (self._w/2.+self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((self._w/2.)*np.sin(theta), (self._w/2.+self._s)*np.sin(theta[::-1])))
        p = PolygonSet(polygons=[np.vstack((x, y)).T],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])

        # generate bounding polygon
        x = np.array(((self._w/2.+self._s)*np.cos(theta[::-1])))
        y = np.array(((self._w/2.+self._s)*np.sin(theta[::-1])))
        bp = PolygonSet(polygons=[np.vstack((x, y)).T])

        if orientation in ('r', 'right'):
            added_ref = (self._w/2 + self._s, 0.)

        elif orientation in ('b', 'bottom'):
            p.rotate(-np.pi/2.)
            bp.rotate(-np.pi/2.)
            added_ref = (0., -self._w/2 - self._s)

        elif orientation in ('l', 'left'):
            p.rotate(np.pi)
            bp.rotate(np.pi)
            added_ref = (-self._w/2 - self._s, 0.)

        elif orientation in ('t', 'top'):
            p.rotate(np.pi/2.)
            bp.rotate(np.pi/2.)
            added_ref = (0., -self._w/2 - self._s)

        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.\nMust be "right", "bottom", "left" or "top".')

        self._add(p.translate(*self.ref))
        self._bounding_polygon+=bp.translate(*self.ref)

        if update_ref:
            self.ref = [self.ref[0]+added_ref[0], self.ref[1]+added_ref[1]]

        return self


    def add_fresnel_end(self, orientation: str,
                              update_ref: bool=False,
                              nb_points: int=101) -> PolygonSet:
        """
        Add a fresnel open end to a coplanar waveguide in the given
        orientation.
        Notice that this method does not increment the total length of the
        waveguide nor its reference point (unless update_ref is set to True)!

        Parameters
        ----------
        orientation : str {'left', right', 'top', 'bottom'}
            Orientation of the open end.
            Also the orientation can be writen in various way, for instance
            following are equivalent:
            'l', 'left'
        update_ref : bool
            If True update the reference point of the cpw by including the length
            of the end in x and y direction.
            Useful when starting a cpw by an end_point.
        nb_point : int (default=101)
            Number of point used in the polygon.
            Must be odd.
        """

        # Get a fresnel curve in the middle of the gap
        t = np.linspace(0, self._get_fresnel_parametric_length(np.pi), nb_points)
        x, y = self._get_fresnel_curve(np.pi, self._w+self._s, nb_points)
        dx, dy = np.gradient(x, t), np.gradient(y, t)

        # Get the external curve
        dx1, dy1 = dx, dy
        n = np.sqrt(dx1**2 + dy1**2.)
        dx1, dy1 = dx1/n, dy1/n
        x1, y1 = x, y
        theta1 = np.angle(dx1+1j*dy1)-np.pi/2.
        x1, y1 = x1+np.cos(theta1)*self._s/2., y1+np.sin(theta1)*self._s/2.

        # Get the internal curve
        dx2, dy2 = dx[::-1], dy[::-1]
        n = np.sqrt(dx2**2 + dy2**2.)
        dx2, dy2 = dx2/n, dy2/n
        x2, y2 = x[::-1], y[::-1]
        theta2 = np.angle(dx2+1j*dy2)+np.pi/2.
        x2, y2 = x2+np.cos(theta2)*self._s/2., y2+np.sin(theta2)*self._s/2.

        # Combine both curve to get a polygon and put its lower left at (0, 0)
        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
        p = PolygonSet(polygons=[np.vstack((x, y)).T],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])
        bp = PolygonSet(polygons=[np.vstack((x2, y2)).T])
        added_length = p.get_size()[0]
        bp.translate(-p.polygons[0][nb_points,0], -p.polygons[0][nb_points,1])
        p.translate(-p.polygons[0][nb_points,0], -p.polygons[0][nb_points,1])

        # Correct points to match perfectly to the width and gap
        p.polygons[0][0] = [0, self._s+self._w]
        p.polygons[0][nb_points-1] = [0, self._s]
        p.polygons[0][nb_points] = [0, 0]
        p.polygons[0][-1] = [0, 2*self._s+self._w]
        p.translate(*self.ref)
        bp.polygons[0][-1] = [0, (2*self._s+self._w)]
        bp.polygons[0][0] = [0, 0]
        bp.translate(*self.ref)

        if orientation in ('r', 'right'):
            added_ref = (added_length, 0)
            self._add(p.translate(0, -self._w/2-self._s))
            self._bounding_polygon+=bp.translate(0, -self._w/2-self._s)

        elif orientation in ('b', 'bottom'):
            p = p.rotate(-np.pi/2, center=p.get_center())
            bp = bp.rotate(-np.pi/2, center=p.get_center())
            dx, dy = (-p.polygons[0][-1,0], -p.polygons[0][-1,1])
            p.translate(dx, dy)
            p.translate(self.ref[0], self.ref[1])
            bp.translate(dx, dy)
            bp.translate(self.ref[0], self.ref[1])

            added_ref = (0, -added_length)

            self._add(p.translate(+self._w/2+self._s, 0))
            self._bounding_polygon+=bp.translate(+self._w/2+self._s, 0)

        elif orientation in ('l', 'left'):
            p = p.flip('y')
            dx, dy = (-p.polygons[0][nb_points,0], -p.polygons[0][nb_points,1])
            p.translate(dx, dy)
            p.translate(self.ref[0], self.ref[1])
            bp.flip('y')
            bp.translate(dx, dy)
            bp.translate(self.ref[0], self.ref[1])

            added_ref = (-added_length, 0)

            self._add(p.translate(0, -self._w/2-self._s))
            self._bounding_polygon+=bp.translate(0, -self._w/2-self._s)

        elif orientation in ('t', 'top'):
            p = p.rotate(np.pi/2, center=p.get_center())
            bp = bp.rotate(np.pi/2, center=p.get_center())
            dx, dy = (-p.polygons[0][-1,0], -p.polygons[0][-1,1])
            p.translate(dx, dy)
            bp.translate(dx, dy)
            p.translate(self.ref[0], self.ref[1])
            bp.translate(self.ref[0], self.ref[1])

            added_ref = (0, added_length)

            self._add(p.translate(-self._w/2-self._s, 0))
            self._bounding_polygon+=bp.translate(-self._w/2-self._s, 0)

        else:
            raise ValueError('Your orientation "'+orientation+'" is incorrect.\nMust be "right", "bottom", "left" or "top".')

        if update_ref:
            self.ref = [self.ref[0]+added_ref[0], self.ref[1]+added_ref[1]]

        return self
