from functools import lru_cache, partial
import numpy as np
from typing import Optional, Union, Tuple
from scipy.special import fresnel
from typing import Literal, Dict, Callable
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from scipy.special import fresnel
from scipy.optimize import fsolve


from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.typing_local import Coordinate



class TransmissionLine(PolygonSet):


    def __init__(
        self,
        layer: int,
        datatype: int,
        name: str,
        color: str,
        ref: Optional[Coordinate] = None,
    ) -> None:
        """
        TransmissionLine class for strip classes allowing easy draw of complex circuits.

        Parameters
        ----------
        layer : int
            Layer number of the coplanar.
        datatype : int
            Datatype number of the coplanar.
        ref : List[float, float] (default None)
            Coordinate of the last points.
            If None, corresponds to [0, 0].
        """

        # Avoid mutability of the default reference by instancing the list inside the object
        if ref is None:
            self.ref: Coordinate = [0., 0.]
        else:
            self.ref = ref

        PolygonSet.__init__(self, polygons=[np.array([self.ref])],
                                  layers=[layer],
                                  datatypes=[datatype],
                                  names=[name],
                                  colors=[color])
        self._layer = layer
        self._datatype = datatype
        self._name = name
        self._color = color

        self.total_length = 0. # Total length of the strip

        # Store computed Fresnel information to save computation timne
        computedSerpentine: Dict[Tuple[float, int, float],
                                 Tuple[np.ndarray, np.ndarray,
                                       np.ndarray,
                                       np.ndarray, np.ndarray,
                                       Callable, Callable]] = {}


    @property
    def layer(self):
        self._layer


    @layer.setter
    def layer(self, layer:int):

        self.layers = [layer]*len(self.polygons)
        self._layer = layer


    @property
    def datatype(self):
        self._datatype


    @datatype.setter
    def datatype(self, datatype:int):

        self.datatypes = [datatype]*len(self.polygons)
        self._datatype = datatype


    @property
    def name(self):
        self._name


    @name.setter
    def name(self, name:str):

        self.names = [name]*len(self.polygons)
        self._name = name


    @property
    def color(self):
        self._color


    @color.setter
    def color(self, color:str):

        self.colors = [color]*len(self.polygons)
        self._color = color


    def _rot(self, x: float,
                   y: float,
                   theta: float) -> Coordinate:
        """
        Rotate set of point by theta in respect to (0, 0).
        """

        tx =  np.cos(theta)*x + np.sin(theta)*y
        ty = -np.sin(theta)*x + np.cos(theta)*y

        return tx, ty


    @partial
    def func(t, args, x, y):
        if t[0]>t[-1]:
            return x, y
        else:
            return x[::-1], y[::-1]


    @partial
    def dfunc(t, args, dx, dy,f_dx, f_dy):
        if isinstance(t, float):
            return f_dx(t), f_dy(t)
        else:
            if t[0]>t[-1]:
                return dx, dy
            else:
                return dx[::-1], dy[::-1]


    @lru_cache
    def calc(self, radius: float,
                   nb_points: int) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray,
                                            np.ndarray, np.ndarray,
                                            Callable, Callable,
                                            float]:

        # Calculate curve length
        def curve_length(t):
            return np.hypot(f_dx(t), f_dy(t))

        t = np.linspace(0, self._get_fresnel_parametric_length(np.pi/2.), nb_points)
        x, y = self._get_fresnel_curve(np.pi/2, radius, nb_points)
        dx, dy = np.gradient(x, t), np.gradient(y, t)

        # We ensure flat derivative to get straight structure
        dx[0], dy[-1] = 0, 0

        f_dx = InterpolatedUnivariateSpline(t, dx)
        f_dy = InterpolatedUnivariateSpline(t, dy)

        length = quad(curve_length, t[0], t[-1])[0]

        return dx, dy, t, x, y, f_dx, f_dy, length



    def translate(self, dx: float,
                        dy: float) -> PolygonSet:
        """
        Translate the polygons by the amount dx, dy in the x and y direction.
        Take care of updating the reference point of the lines.
        Take care of updating the parametric curve of the lines.

        Args:
            dx (float): amount of translation in the x direction in um.
            dy (float): amount of translation in the y direction in um.

        Returns:
            PolygonSet: translated polygons.
        """

        self.ref[0] += dx
        self.ref[1] += dy

        if hasattr(self, '_param_curve'):
            self._param_curve[:,0] += dx
            self._param_curve[:,1] += dy

        return super().translate(dx, dy)


    ###########################################################################
    #
    #                   Add polygons to the existing polygons
    #
    ###########################################################################


    def _add(self, r: PolygonSet) -> None:
        """
        Merge polygons to existing microstrip line.
        """

        self += r


    ###########################################################################
    #
    #                   Add points to internal parametric curve
    #
    ###########################################################################


    def _add2param(self, x: Union[np.ndarray, list],
                         y: Union[np.ndarray, list],
                         t: Union[np.ndarray, list]) -> None:
        """
        Add points to internal parametric curve

        Args:
            x: x coordinate in um
            y: y coordinate in um
            t: distance along the parametric curve in um
        """



        if not hasattr(self, '_param_curve'):
            self._param_curve = np.vstack((x, y, t)).T
        else:
            new_values = np.array([x, y, self._param_curve[:,2][-1]+t]).T

            self._param_curve = np.concatenate((self._param_curve, new_values), axis=0)


    ###########################################################################
    #
    #                         Fresnel turn
    #
    ###########################################################################


    def _get_fresnel_parametric_length(self, angle: float) -> float:
        """
        Return the parametric length of the curve corresponding to half the
        given angle.

        Parameters
        ----------
        angle : float
            Angle in radian
        """

        # Pre-computer value for standard angle
        # This angle is used a lot to compute fresnel turn
        if angle==np.pi/2:
            return 0.710657090720168
        else:
            def func(t0: float,
                    target_angle: float) -> float:

                # Get the x coordinate corresponding to some parametric length
                y1, x1 = fresnel(t0)
                y0, x0 = fresnel(t0*0.99)

                return target_angle - np.angle((x1-x0) + 1j*(y1-y0))

            # Get the first half of the Fresnel curve going to 0deg to half of the
            # desired angle
            return fsolve(func, 0.75, args=np.abs(angle)/2.)[0]


    def _get_fresnel_curve(self, angle: float,
                                 radius: float,
                                 nb_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the parametric Fresnel curve along the length t.
        The curve is doubled, mirrored, joint and normalized to a given radius

        Parameters
        ----------
        angle : float
            Total angle of the curve in radian
            Must be pi or pi/2
        radius : float
            Radius of the curve
        nb_points : int
                Number of point in the polygon
        """

        if angle!=np.pi and angle!=np.pi/2:
            raise ValueError('Fresnel curve are only availabe for 90 and 180 degrees turn.')
        if nb_points%2==0:
            raise ValueError('nb_points must be an odd number.')

        t0 = self._get_fresnel_parametric_length(angle)
        t = np.linspace(0, t0, int(nb_points/2)+1)

        y1, x1 = fresnel(t)

        # Get the second half that we have to move and snap to the first one
        # Center
        x2 = x1 - x1.max()
        y2 = y1 - y1.max()

        # Mirror symetry on y
        x2 = +x2
        y2 = -y2

        # Rotation
        if angle==np.pi/2:
            x2a, y2a = self._rot(x2, y2, np.pi/2)
        else:
            x2a, y2a = x2, y2

        # Translation
        x2a += x1[-1]
        y2a += y1[-1]

        # Assembling
        y = np.concatenate((y1[:-1], y2a))
        x = np.concatenate((x1[:-1], x2a))

        # Sorting along the y direction
        y_t = np.argsort(y)
        y = y[y_t]
        x = x[y_t]

        # Normalize to the correct radius

        # Get the longest direction
        ym = abs(y.max()-y.min())
        xm = abs(x.max()-x.min())

        if xm>=ym:
            r = xm
        else:
            r = ym

        y = y/r*radius+self._w/2.
        x = x/r*radius

        return x[::-1], y[::-1]


    ###########################################################################
    #
    #                   Meta method to create complex microstrip path
    #
    ###########################################################################


    def add_serpentine(self, total_length: float,
                             nb_turn: int,
                             spacing: float,
                             orientation: Literal['right', 'left', 'top', 'bottom']='right',
                             starting: Literal['left', 'right']='left',
                             spacing_mode: Literal['center2center', 'edge2edge']='center2center',
                             turn: Literal['circular', 'fresnel']='circular',
                             nb_points_per_turn: Optional[int]=None) -> PolygonSet:
        """
        Add a microstrip serpentine.
        The spacing between the strip can be calculated from strip-center to
        strip-center or strip-edge to strip-edge through the spacing_mode
        parameter

        Parameters
        ----------
        total_length : float
            Total length of the serpentine in um.
        nb_turn : int
            Number of serpentine turn.
        spacing : float
            Spacing between the strip in um.
            This spacing is calculated through two different modes, see
            spacing_mode.
        orientation : str {'right', 'left', 'top', 'bottom'} (default 'right')
            Orientation of the serpentine.
        starting : str {'right', 'left'} (default 'left')
            Orientation of the first turn of the serpentine in respect to the
            last microstrip segment.
        spacing_mode : str {'center2center', 'edge2edge'}, (default 'center2center')
            The spacing between the strip can be calculated from strip-center to
            strip-center or strip-edge to strip-edge depending of this parameter.
        turn : str {'circular', 'fresnel'}, (default 'circular')
            Type or turn.
        nb_points_per_turn : Optional(int) (default None)
            Number of point used per turn.
            If None, used the default number of point of the circular, 50, or
            Fresnel, 51, turn.
        """

        # Calculate the radius depending of the spacing mode
        if spacing_mode == 'center2center':
            r = spacing/2.
        elif spacing_mode == 'edge2edge':
            r = spacing/2. + self._w/2.
        else:
            raise ValueError('"spacing_mode" must be in ["center2center", "edge2edge"]')

        # Depending of the turn mode, the turn function and the turn length vary
        if turn not in ('circular', 'fresnel'):
            raise ValueError('"turn" must be in ["circular", "fresnel"]')
        else:
            if turn=='circular':
                if nb_points_per_turn is None:
                    nb_points_per_turn = 50

                turn_length =  np.pi*r/2
                turn_func = self.add_turn
            elif turn=='fresnel':
                if nb_points_per_turn is None:
                    nb_points_per_turn = 51

                # All the computations are stored in self.calc to cache the result
                dx, dy, t, x, y, f_dx, f_dy, turn_length = self.calc(r, 1001)

                turn_func = self.add_fresnel_turn

        # Calculate the serpentine length
        s_length = (total_length-(2*nb_turn+2)*turn_length+2*r)/2/nb_turn

        # Prepare variable to build serpentine depending on its orientation
        if orientation.lower() in ['right', 'left']:
            c = 'b'
            d = 't'
            x1 = 0.
            x2 = 0.
            y1 = s_length-r
            y2 = 2.*s_length

            if orientation.lower() == 'right':
                a = 'l'
                b = 'r'
            elif orientation.lower() == 'left':
                a = 'r'
                b = 'l'
                if starting.lower()=='left':
                    starting = 'right'
                else:
                    starting = 'left'
        elif orientation.lower() in ['top', 'bottom']:
            c = 'l'
            d = 'r'
            x1 = s_length-r
            x2 = 2.*s_length
            y1 = 0.
            y2 = 0.
            if orientation.lower()=='top':
                a = 'b'
                b = 't'
                if starting.lower()=='left':
                    starting = 'right'
                else:
                    starting = 'left'
            elif orientation.lower()=='bottom':
                a = 't'
                b = 'b'
        else:
            raise ValueError('"orientation" must be "right", "left", "top", "right".')

        # Start
        if starting.lower()=='left':
            turn_func(r, a+d, nb_points=nb_points_per_turn)
            self.add_line(x1, y1)
            counter = 0
            final = nb_turn - 1
        elif starting.lower()=='right':
            turn_func(r, a+c, nb_points=nb_points_per_turn)
            self.add_line(-x1, -y1)
            counter = 1
            final = nb_turn
        else:
            raise ValueError('starting must be "left" or "right".')

        # Middle
        while counter<final:
            if counter%2:
                turn_func(r, d+b, nb_points=nb_points_per_turn)
                turn_func(r, a+d, nb_points=nb_points_per_turn)
                self.add_line(x2, y2)
            else:
                turn_func(r, c+b, nb_points=nb_points_per_turn)
                turn_func(r, a+c, nb_points=nb_points_per_turn)
                self.add_line(-x2, -y2)

            counter += 1

        # End from either to or bottom
        if counter%2:
            turn_func(r, d+b, nb_points=nb_points_per_turn)
            turn_func(r, a+d, nb_points=nb_points_per_turn)
            self.add_line(x1, y1)
            turn_func(r, c+b, nb_points=nb_points_per_turn)
        else:
            turn_func(r, c+b, nb_points=nb_points_per_turn)
            turn_func(r, a+c, nb_points=nb_points_per_turn)
            self.add_line(-x1, -y1)
            turn_func(r, d+b, nb_points=nb_points_per_turn)

        return self

