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


class MicroStrip_Polar(TransmissionLine):

    def __init__(self, width: float,
                       angle: float,
                       layer: int=0,
                       datatype: int=0,
                       name: str='',
                       color: str='')-> None:
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
                                        color=color)

        self._w = width
        self._angle=angle


    @property
    def width(self):
        return self._w
    def angle(self):
        return self._angle


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

    def add_line(self, l_len: float,
                                        ) -> PolygonSet:
        """
        Add a piece of linear microstrip in the x or y direction .

        Parameters
        ----------
        x : float
            Length of the strip in the x direction in um.
        y : float
            Length of the strip in the y direction in um.
        """
        r = Rectangle((self.ref[0], self.ref[1]-self._w/2),
                              (self.ref[0]+l_len, self.ref[1]+self._w/2),
                              layer=self._layer,
                              datatype=self._datatype,
                              name=self._name,
                              color=self._color).rotate(self._angle,[self.ref[0],self.ref[1]])
        self._add2param(self._rot(self.ref[0],self.ref[1]+self._w/2,-self._angle),
                        self._rot(self.ref[0]+l_len, self.ref[1]+self._w/2,-self._angle),
                        [0, abs(l_len)])         

        self._add(r)
        self.ref = [self.ref[0] + self._rot(l_len, 0,-1*self._angle)[
            0], self.ref[1] + self._rot(l_len, 0,-1*self._angle)[1]]
        self.total_length += abs(l_len)

        return self


    ###########################################################################
    #
    #                       Add turn
    #
    ###########################################################################


    def add_turn(self, radius: float,
                       delta_angle: float,
                       nb_points: int=50) -> PolygonSet:

        if delta_angle >= 0:
            start= self._angle - np.pi/2
            stop= start + delta_angle 
            theta=np.linspace(start,stop, nb_points)
            x0 = self.ref[0] + -radius*np.cos(start) 
            y0 = self.ref[1] + -radius*np.sin(start) 
            x = np.concatenate(((radius-self._w/2.)*np.cos(theta), (radius + self._w/2.)*np.cos(theta[::-1])))
            y = np.concatenate(((radius-self._w/2.)*np.sin(theta), (radius + self._w/2.)*np.sin(theta[::-1])))
            self.ref = [x0+(x[nb_points]+x[nb_points-1])/2, y0+(y[nb_points]+y[nb_points-1])/2] 
            
        else:
            start= self._angle + np.pi/2
            stop= start + delta_angle 
            theta=np.linspace(start,stop, nb_points)
            x0 = self.ref[0] + -radius*np.cos(start) 
            y0 = self.ref[1] + -radius*np.sin(start) 
            x = np.concatenate(((radius-self._w/2.)*np.cos(theta), (radius + self._w/2.)*np.cos(theta[::-1])))
            y = np.concatenate(((radius-self._w/2.)*np.sin(theta), (radius + self._w/2.)*np.sin(theta[::-1])))
            self.ref = [x0+(x[nb_points]+x[nb_points-1])/2, y0+(y[nb_points]+y[nb_points-1])/2]  
        
        
        self._angle += delta_angle

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


    ###########################################################################
    #
    #                               Tapers
    #
    ###########################################################################

    def add_taper(self, l_len: float,
                        
                        new_width: float) -> PolygonSet:

        p = [(self.ref[0], self.ref[1]-self._w/2),
             (self.ref[0]+l_len, self.ref[1] - new_width/2.),
             (self.ref[0]+l_len, self.ref[1] + new_width/2.),
             (self.ref[0], self.ref[1]+self._w/2)]

        self._add2param([self.ref[0], self.ref[0]+l_len],
                        [self.ref[1]+self._w/2, self.ref[1]+self._w/2],
                        [0, abs(l_len)])

        self._add(PolygonSet(polygons=[p],
                             layers=[self._layer],
                             datatypes=[self._datatype],
                             names=[self._name],
                             colors=[self._color]).rotate(self._angle,[self.ref[0],self.ref[1]]))
        #self.ref = [self.ref[0]+l_len, self.ref[1]]
        self.ref = [self.ref[0] + self._rot(l_len , 0,-self._angle)[0], self.ref[1] + self._rot(l_len , 0, -self._angle)[1]]
        self.total_length += abs(l_len)
        
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
                          colors=[self._color]).translate(self.ref[0]-x[0]+self._rot(self._w/2,0,-theta1[0])[0], self.ref[1]-y[0]+self._rot(self._w/2,0,-theta1[0])[1]).rotate(self._angle - theta1[0] - np.pi/2, [self.ref[0], self.ref[1]])
        self._angle += theta1[-1] - theta1[0]
        self.ref[0]+= 10
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
            self.ref = (poly.polygons[0][int(len(poly.polygons[0])/2)-1] + poly.polygons[0][int(len(poly.polygons[0])/2)])/2
            self._add(poly)

            return self
        else:
            return poly
    