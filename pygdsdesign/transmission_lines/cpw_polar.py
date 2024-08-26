import numpy as np
import math
import copy
from typing import Callable, Tuple, Optional, Union
from scipy.integrate import quad

from pygdsdesign.polygons import Rectangle
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.transmission_lines.transmission_line import TransmissionLine
from pygdsdesign.typing_local import Coordinate
from pygdsdesign.shapes import butterworth_filter

class CPWPolar(TransmissionLine):

    def __init__(
        self,
        width: float,
        gap: float,
        angle: float,
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
        angle: float
            Orientation of the microstrip in radian.
            This angle can be modified latter with the add_turn function.
            A value of 0 corresponds to the direction left to right.
        layer : int
            Layer number of the coplanar. Default to 0
        datatype : int
            Datatype number of the coplanar. Default to 0
        name: str
            Name of the complanar
        color: str
            Color of the complanar
        """

        TransmissionLine.__init__(self, layer=layer,
                                        datatype=datatype,
                                        name=name,
                                        color=color,
                                        ref=ref)

        self._w = width
        self._angle = angle
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

    ###########################################################################
    #
    #                   Add polygons to the existing coplanar waveguide
    #
    ###########################################################################


    def add_line(self, l_len: float) -> PolygonSet:

        """
        Add a piece of linear coplanar in the direction of the angle.

        Parameters
        ----------
        l_len : float
            Length of the strip in um.
        """
        p  = PolygonSet([[(0, -self._w/2.),
                          (0, -self._w/2. - self._s),
                          (l_len, -self._w/2. - self._s),
                          (l_len, -self._w/2.)]],
                         layers=[self._layer],
                         datatypes=[self._datatype],
                         names=[self._name],
                         colors=[self._color])

        p += PolygonSet([[(0, +self._w/2.),
                            (0, +self._w/2. + self._s),
                            (l_len, +self._w/2. + self._s),
                            (l_len, +self._w/2.)]],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color])

        a, b = p.get_bounding_box()
        self._add(p.rotate(self._angle).translate(*self.ref))

        # update bounding polygon

        bp = Rectangle((a[0], a[1]),
                       (b[0], b[1])).rotate(self._angle).translate(*self.ref)

        self.ref = [self.ref[0] + l_len* \
                    np.cos(1*self._angle), self.ref[1] + l_len*np.sin(1*self._angle)]

        self.total_length += abs(l_len)
        self._bounding_polygon += bp

        return self


    ###########################################################################
    #
    #                       Add turn
    #
    ###########################################################################


    def add_turn(self, radius: float,
                       delta_angle: float,
                       nb_points: int=50) -> PolygonSet:
        """
        Add a circulare turn to the strip.

        Parameters
        ----------
        radius : float
            Radius of the arc in um.
        delta_angle : float
            Angle of the turn. a positive value will produces a left turn. A
            A negative value will produces a right turn.
            The angle is relative to the previous angle.
            Hence, a value of pi/2 will produces a 90° left turn,
            relatives to the direction of the last strip.
        nb_point : int (default=50)
            Number of point used in the polygon.
        """

        if delta_angle >= 0:
            start= self._angle - np.pi/2
        else:
            start= self._angle + np.pi/2
        stop= start + delta_angle
        theta=np.linspace(start,stop, nb_points)

        x0 = self.ref[0] + -radius*np.cos(start)
        y0 = self.ref[1] + -radius*np.sin(start)

        x = np.concatenate(((radius+self._w/2.)*np.cos(theta), (radius+self._w/2.+self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius+self._w/2.)*np.sin(theta), (radius+self._w/2.+self._s)*np.sin(theta[::-1])))

        p = PolygonSet(polygons=[np.vstack((x+x0, y+y0)).T],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])

        x = np.concatenate(((radius-self._w/2.)*np.cos(theta), (radius-self._w/2.-self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius-self._w/2.)*np.sin(theta), (radius-self._w/2.-self._s)*np.sin(theta[::-1])))

        p += PolygonSet(polygons=[np.vstack((x+x0, y+y0)).T],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color])

        # generate bounding polygon
        x = np.concatenate(((radius+self._w/2.+self._s)*np.cos(theta), (radius-self._w/2.-self._s)*np.cos(theta[::-1])))
        y = np.concatenate(((radius+self._w/2.+self._s)*np.sin(theta), (radius-self._w/2.-self._s)*np.sin(theta[::-1])))

        bp = PolygonSet(polygons=[np.vstack((x+x0, y+y0)).T])

        self._angle += delta_angle
        self.ref = [x0+(x[nb_points]+x[nb_points-1])/2,
                    y0+(y[nb_points]+y[nb_points-1])/2]
        self._add(p)
        self._bounding_polygon+=bp
        self.total_length += radius*delta_angle

        return self



    ###########################################################################
    #
    #                               Tapers
    #
    ###########################################################################


    def add_taper(self, l_len: float,
                        new_width: float,
                        new_gap: float) -> PolygonSet:
        """
        Add linear taper between the current and the new width.

        Parameters
        ----------
        l_len : float
            Length of the taper in um.
        new_width : float
            New width of the microstrip in um.
        new_gap : float
            New gap of the microstip in um.
        """

        p = PolygonSet(polygons=[[(0., self._w/2.)]],
                       layers=[self._layer],
                       datatypes=[self._datatype],
                       names=[self._name],
                       colors=[self._color])
        p>(l_len, new_width/2.-self._w/2.)
        p>(0., new_gap)
        p>(-l_len, -new_gap-new_width/2.+self._s+self._w/2.)
        p += copy.copy(p).mirror((0, 0), (1, 0))
        p.rotate(self._angle)

        # generate bounding polygon
        bp = PolygonSet(polygons=[[(0, self._w/2+self._s),
                                  (l_len, new_width/2+new_gap),
                                  (l_len, 0),
                                  (0, 0)]]
                          )
        bp += copy.copy(bp).mirror((0, 0), (1, 0))

        p.translate(*self.ref)
        bp.translate(*self.ref).rotate(self._angle,[self.ref[0],self.ref[1]])

        self.ref = [self.ref[0]+l_len*np.cos(self._angle), self.ref[1]+l_len*np.sin(self._angle)]

        self._add(p)
        self._bounding_polygon+=bp

        self.total_length += abs(l_len)
        self._w = new_width
        self._s = new_gap

        return self

    ###########################################################################
    #
    #                   Generic parametric curve
    #
    ###########################################################################


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
        The curve is automtically aligned and rotated according to the previous strip angle.
        The next strip's angle will also be changed according to the curve.

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
                       colors=[self._color]).translate(self.ref[0]-x[0]+self._rot(-self._w/2-self._s,0,-theta1[0])[0], self.ref[1]-y[0]+self._rot(-self._w/2-self._s,0,-theta1[0])[1]).rotate(self._angle - theta1[0] - np.pi/2, [self.ref[0], self.ref[1]])\
           + PolygonSet(polygons=[p2],
                        layers=[self._layer],
                        datatypes=[self._datatype],
                        names=[self._name],
                        colors=[self._color]).translate(self.ref[0]-x[0]+self._rot(-self._w/2-self._s,0,-theta1[0])[0], self.ref[1]-y[0]+self._rot(-self._w/2-self._s,0,-theta1[0])[1]).rotate(self._angle - theta1[0] - np.pi/2, [self.ref[0], self.ref[1]])
        # generate bounding_polygon
        bp = PolygonSet(polygons=[bp])
        bp = bp.translate(self.ref[0]-x[0] - (self._w/2 + self._s)*np.cos(theta1[0]), self.ref[1]-y[0] - (self._w/2 + self._s)*np.sin(theta1[0])
                          ).rotate(self._angle-theta1[0]-np.pi/2, [self.ref[0], self.ref[1]])
        self._angle += theta1[-1] - theta1[0]
        # Calculate curve length
        def func(t, args):
            dx1, dy1 = df(t, args)
            return np.hypot(dx1, dy1)

        # Add the length of the parametric curve only if asked (default)
        if add_length:
            self.total_length += quad(func, t[0], t[-1], args=(args,))[0]

        # Add polygon only if asked (default)
        if add_polygon:
            self.ref = [p.polygons[0][int(len(p.polygons[0])/2)][0] -self._w/2*np.cos(self._angle-np.pi/2),
                        p.polygons[0][int(len(p.polygons[0])/2)][1] - self._w/2*np.sin(self._angle-np.pi/2)]
            self._add(p)
            self._bounding_polygon+=bp

            return self
        else:
            return p, bp

    ###########################################################################
    #
    #                             Ends
    #
    ###########################################################################

    def add_end(self, width: float,
                      update_ref: bool=False) -> PolygonSet:
        """
        Add an end to a coplanar waveguide in the perpendicular direction

        Parameters
        ----------
        width : float
            width of the end in um
        """

        r = Rectangle((-self._w/2.-self._s, width),
                      (self._w/2.+self._s, 0),
                          layer=self._layer,
                          datatype=self._datatype,
                          name=self._name,
                          color=self._color)
        a,b = r.get_bounding_box()
        self._add(r.translate(*self.ref).rotate(self._angle-np.pi/2,[self.ref[0],self.ref[1]]))


        if update_ref:
            self.ref = [self.ref[0]+width*np.cos(self._angle), self.ref[1]+width*np.sin(self._angle)]

        # update bounding polygon
        bp = Rectangle((a[0], a[1]),
                        (b[0], b[1])).translate(*self.ref).rotate(self._angle-np.pi/2,[self.ref[0],self.ref[1]])
        self._bounding_polygon+=bp

        return self


    def add_circular_end(self, nb_points: int=50,
                               update_ref: bool=False) -> PolygonSet:
        """
        Add a circular open end to a coplanar waveguide in the given
        orientation.

        Parameters
        ----------
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

        p.rotate(self._angle)
        bp.rotate(self._angle)
        self._add(p.translate(*self.ref))
        self._bounding_polygon+=bp.translate(*self.ref)

        if update_ref:
            added_ref = self._rot(self._w/2 + self._s, 0,self._angle)
            self.ref = [self.ref[0]+added_ref[0], self.ref[1]-added_ref[1]]
        return self


    ###########################################################################
    #
    #                             misc
    #
    ###########################################################################

    def add_butterworth_filter(self,central_conductor_width=4,
                       central_conductor_gap=0.2,
                       sep_bot_top=6,
                       sep_antenna_indutance=6,
                       sep_inductance_antenna=6,
                       sep_antenna_central=2,#
                       nb_l_horizontal=1,
                       len_l_horizontal=70,
                       len_l_vertical=5,
                       l_microstrip_width=0.5,#
                       c_arm_length1=[18,26,34,34,26,18],
                       c_arm_length2=[18,28,38,38,28,18],
                       c_arm_length3=[34,26,18],
                       c_central_width=2,
                       arm_width=2,
                       gap=0.2,
                       length=[14,32],
                       ):
        """
        Create a butterworth filter and its bouding polygons.

        Args:
            central_conductor_width: width of the central CPW.
                Defaults to 4 um
            central_conductor_gap: gap of the central CPW.
                Defaults to 0.2um
            sep_bot_top: distance added at the start and at the end of the filter. used to separate the filter from other elements.
                Defaults to 6um.
            sep_antenna_indutance: distance between the first capacitance and the first indutance, and between the second inductance and the third capacitance.
                Defaults to 6um.
            sep_inductance_antenna: distance between the first inductance and the second capacitance and between the second capacitance and the second inductance.
                Defaults to 6um.
            sep_antenna_central: distance between the capacitance and the central conductor.
                Defaults to 6um.
            nb_l_horizontal: inductance parameters. number of time the microstrip will go from one side to an other, the first and the last half-length microstrip dont count.
                Defaults to 1.
            len_l_horizontal: inductance parameters. length of the microstrip going from the left (or right) to the right (or left) side.
                Defaults to 70 um.
            len_l_vertical: inductance parameters. distance between two horizontal microstrip.
                Defaults to 5 um.
            l_microstrip_width: inductance parameters. width of the microstrip.
                Defaults to 0.5um.
            c_arm_length1/2/3: capacitance parameters. length of the arm of the first and third capacitance/ the first and the third part of the second capacitance / the second part of the second capacitance.
                Defaults to [18,24,36,24,18] [um].
            c_central_width: capacitance parameters. width of the central conductor, connecting the arms.
                Defaults to 2 um.
            arm_width: capacitance parameters. width of the arms.
                Defaults to 2 um.
            gap: capacitance parameters. distance between the the conductor and the ground plane.
                Defaults to 0.2 um.
            length: capacitance parameters. lengths of the central conductor of the capacitance, connecting the arms.
                Defaults to [14,32]um.
            layer,datatype,name,color: Used for naming the metal layer
                Defaults to 0,0,'',''.
        """
        b_filter=PolygonSet()
        bp=PolygonSet()
        a,b= butterworth_filter(central_conductor_width,central_conductor_gap,sep_bot_top,sep_antenna_indutance,sep_inductance_antenna,sep_antenna_central,nb_l_horizontal,len_l_horizontal,len_l_vertical,l_microstrip_width,c_arm_length1,c_arm_length2,c_arm_length3,c_central_width,arm_width,gap,length,
                          layer=self._layer,
                          datatype=self._datatype,
                          name=self._name,
                          color=self._color)

        b_filter+=a
        bp+=b

        d= b_filter.get_size()[1]
        p=b_filter.rotate(self._angle-np.pi/2)
        bp=bp.rotate(self._angle-np.pi/2)

        self._bounding_polygon += bp.translate(*self.ref).change_layer(1)
        self._add(p.translate(*self.ref))
        self.ref = [self.ref[0]+ np.cos(self._angle) * d, self.ref[1]+ np.sin(self._angle) * d]
        return self


    def goto(self, p_destination):
        """
    Create a link between two CPWs.

    Args:
        p_destination: CPWPolar instance of the destination.
        """
        def distance(x1:float,y1:float,x2:float,y2:float) -> float:
            """Return the distance between the two point (x1;y1) and (x2;y2)"""
            return np.sqrt((x1-x2)**2 + (y2-y1)**2)

        largeur= self._w/2 + self._s
        ref2=p_destination.ref
        angle2=(p_destination._angle+np.pi) %(np.sign(p_destination._angle+1e-9)*2*np.pi)
        if angle2 > np.pi:
            angle2-=2*np.pi
        if angle2 < -np.pi:
            angle2+=2*np.pi

        ref1=self.ref

        angle1=self._angle %(np.sign(self._angle+1e-9)*2*np.pi)
        if angle1 > np.pi:
            angle1-=2*np.pi
        if angle1 < -np.pi:
            angle1+=2*np.pi

        x1=(ref1[0]+np.cos(angle1-np.pi/2)*largeur)
        y1=(ref1[1]+np.sin(angle1-np.pi/2)*largeur)

        x2=(ref1[0]-np.cos(angle1-np.pi/2)*largeur)
        y2=(ref1[1]-np.sin(angle1-np.pi/2)*largeur)


        x3=ref2[0]+np.cos(angle2-np.pi/2)*largeur
        y3=ref2[1]+np.sin(angle2-np.pi/2)*largeur

        x4=ref2[0]-np.cos(angle2-np.pi/2)*largeur
        y4=ref2[1]-np.sin(angle2-np.pi/2)*largeur


        xy=[distance(x1,y1,x3,y3),distance(x1,y1,x4,y4),distance(x2,y2,x3,y3),distance(x2,y2,x4,y4)]
        xy_min=min(xy)
        min_= np.where(xy==xy_min)[0][0]

        if min_==0:
            dx=x3-x1
            dy=y3-y1
        if min_==1:
            dx=x4-x1
            dy=y4-y1
        if min_==2:
            dx=x3-x2
            dy=y3-y2
        if min_==3:
            dx=x4-x2
            dy=y4-y2

        if min_ == 0 or min_ ==3:
            x=math.atan2(dy,dx)%(2*np.pi)

            if x > np.pi:
                x-=2*np.pi
            if x < -np.pi:
                x+=2*np.pi

            turn1 = (x - angle1)

            if turn1 > np.pi:
                turn1-=2*np.pi
            if turn1 < -np.pi:
                turn1+=2*np.pi

            self.add_turn(largeur, turn1)
            self.add_line(xy_min)

            turn2=(angle2-x)

            if turn2 > np.pi:
                turn2-=2*np.pi
            if turn2 < -np.pi:
                turn2+=2*np.pi

            self.add_turn(largeur, turn2)

        else:
            if min_==2:
                self.add_turn(largeur,np.pi/2)
            if min_==1:
                self.add_turn(largeur,-np.pi/2)
            self.goto(p_destination)
        return self