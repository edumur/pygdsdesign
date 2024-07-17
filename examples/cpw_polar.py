import numpy as np
from copy import copy,deepcopy

from pygdsdesign import (GdsLibrary, PolygonSet, Text, CPWPolar)

def spiral(u, args):

    r = args[1]*2.*args[2]*u + args[0]/2.
    theta = args[1]*2.*np.pi*u
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def dspiral_dt(u, args):

    r = args[1]*2.*args[2]*u
    theta = args[1]*2.*np.pi*u
    dx_dt = -args[1]*2.*np.pi * \
        np.sin(theta)*r + np.cos(theta)*args[1]*2.*args[2]
    dy_dt = args[1]*2.*np.pi*np.cos(theta)*r + np.sin(theta)*args[1]*2.*args[2]
    return dx_dt, dy_dt


def circle(t, args):
    x = args[0] * np.cos(t)
    y = args[0] * np.sin(t)
    return x, y

def dcircle_dt(t, args):
    dx_dt = - args[0] * np.sin(t)
    dy_dt = + args[0] * np.cos(t)
    return dx_dt, dy_dt


# Chip size
chip_dx = 595*2
chip_dy = 767*2

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')


# Create empty polygon to which other polygon will be added
tot = PolygonSet()


# Add label on the top right of the chip
t = Text('CPW_Polar', 250)
dx_t, dy_t = t.get_size()
tot += t.translate(-dx_t+chip_dx, -dy_t-250)



# Center the design to the chip and the chip to the (0,0) of the cell

largeur_micro=1
m = CPWPolar(width=largeur_micro, gap=largeur_micro/5, angle=-np.pi/2.24654)

m.add_line(3)

u = np.linspace(0, 1, 10000)
inner_diameter = 10
nb_turn = 4
spiral_spacing = 3.
spiral_width = 2.
m.add_parametric_curve(spiral, dspiral_dt, u, args=(inner_diameter, nb_turn, spiral_spacing))

m.add_line(1)
m.add_line(3)
m.add_line(1)
m.add_line(3)
m.add_turn(3, np.pi/3)
m.add_line(3)
m.add_turn(3, -np.pi/2.56)
m.add_line(3)
m.add_taper(3, largeur_micro*2, largeur_micro/2)
m.add_turn(3, -np.pi/2.56)
m.add_line(3)
m.add_turn(3, np.pi/2.56)
m.add_taper(3, largeur_micro, largeur_micro/5)
m.add_turn(3, -np.pi/2.56)
m.add_line(3)
m.add_parametric_curve(circle, dcircle_dt,np.linspace(-np.pi/2.24, np.pi/0.55412 , 100), args=[0.732854],add_polygon=True)
m.add_line(3)
m.add_turn(3, -np.pi/0.512)
m.add_turn(3, +np.pi/1.2124)
m.add_line(3)
m.add_parametric_curve(circle, dcircle_dt, np.linspace(-np.pi /2.24, np.pi/1.56451, 100), args=[1.15], add_polygon=True)
m.add_line(3)
m.add_parametric_curve(circle, dcircle_dt,np.linspace(-np.pi/2.24, np.pi/0.55412 , 100), args=[0.732854],add_polygon=True)
m.add_line(3)
m.add_parametric_curve(circle, dcircle_dt,np.linspace(-np.pi/3.65, np.pi/2.55412 , 100), args=[1.4854],add_polygon=True)
m.add_taper(3, largeur_micro/2, largeur_micro/10)
m.add_end(0.02, True)
m.add_line(3)
m.add_turn(3, +np.pi/2.2124)
m.add_circular_end(nb_points=101, update_ref=True)
m.add_line(3)

#adding bounding polygon
b_p = CPWPolar(width=largeur_micro, gap=largeur_micro/5, angle=-np.pi/2.24654,layer=42)
b_p+=m.bounding_polygon.change_layer(42)
cell.add(b_p)
tot+=m

tot+=m




cell.add(tot)
# Save GDS file
lib.write_gds('cpw_polar.gds')