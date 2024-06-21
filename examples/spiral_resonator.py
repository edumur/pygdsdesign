import numpy as np
import copy
import time

from pygdsdesign import GdsLibrary, PolygonSet, Rectangle, Text, MicroStrip

start = time.time()

# Chip size of 5mm x 7mm
chip_dx = 5000
chip_dy = 7000

layer = {'nb':1, 'name': 'Al'}

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')


# Create empty polygon to which other polygon will be added
tot = PolygonSet()

# Chip mark
p = Rectangle((0,0), (200, 10), layer=layer['nb'], name=layer['name']).center()
p += Rectangle((0,0), (10, 200), layer=layer['nb'], name=layer['name']).center()
dx_d, dy_d = p.get_size()
tot += p
tot += copy.copy(p).translate(chip_dx, 0)
tot += copy.copy(p).translate(chip_dx, -chip_dy)
tot += copy.copy(p).translate(0., -chip_dy)


# Create design polygon which will contain one feedline and one resonator
p = PolygonSet()

# Feedline
m = MicroStrip(width=500, layer=layer['nb'], name=layer['name'])
m.add_line(500, 0)
m.add_taper_cosec(100, 0, 10, 10)
m.add_line(chip_dx-(500+100)*2., 0)
m.add_taper_cosec(100, 0, 500, 10)
m.add_line(500, 0)
p += m
dx_f, dy_f = m.get_size()


# Resonator
def spiral(u, args):

    r = args[1]*2.*args[2]*u + args[0]/2.
    theta = args[1]*2.*np.pi*u
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def dspiral_dt(u, args):

    r = args[1]*2.*args[2]*u
    theta = args[1]*2.*np.pi*u
    dx_dt = -args[1]*2.*np.pi*np.sin(theta)*r + np.cos(theta)*args[1]*2.*args[2]
    dy_dt =  args[1]*2.*np.pi*np.cos(theta)*r + np.sin(theta)*args[1]*2.*args[2]
    return dx_dt, dy_dt


t = np.linspace(0, 1, 10000)
inner_diameter = 10
nb_turn = 40
spiral_spacing = 3.
spiral_width = 2.

m = MicroStrip(width=spiral_width, layer=layer['nb'], name=layer['name'])
m.add_parametric_curve(spiral, dspiral_dt, t,
                                args=(inner_diameter, nb_turn, spiral_spacing))
dx_s, dy_s = m.get_size()
p += m.translate(dx_f/2.-dx_s/2., dy_s/2.+dy_f/2.+8.5)


# Copy the feedline and the resonator 4 times
p += copy.copy(p).translate(0, -1500)
p += copy.copy(p).translate(0, -3000)


# Add label on the top right of the chip
t = Text('R0C0', 250, layer=layer['nb'], name=layer['name'])
dx_t, dy_t = t.get_size()
tot += t.translate(-dx_t+chip_dx-250, -dy_t-250)


# Center the design to the chip and the chip to the (0,0) of the cell
tot.center()
tot += p.center()

# Add polygons to cell
cell.add(tot)

# Save GDS file
lib.export_gds('spiral_resonator.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')
