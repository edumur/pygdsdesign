import numpy as np
import copy

from pygdsdesign import GdsLibrary, PolygonSet, Rectangle, Text, grid_cover, CPW

# Chip size of 5mm x 7mm
chip_dx = 5000
chip_dy = 7000

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')


# Create empty polygon to which other polygon will be added
tot = PolygonSet()

# Chip mark
p = Rectangle((0,0), (200, 10), layer=0).center()
p += Rectangle((0,0), (10, 200), layer=0).center()
dx_d, dy_d = p.get_size()
tot += p
tot += copy.deepcopy(p).translate(chip_dx, 0)
tot += copy.deepcopy(p).translate(chip_dx, -chip_dy)
tot += copy.copy(p).translate(0., -chip_dy)


# Create design polygon which will contain one feedline and one resonator
p = PolygonSet()

# Feedline
m = CPW(width=500, gap=100, layer=0)
m.add_end(100, 0)
m.add_line(400, 0)
m.add_taper(300, 0, 4, 2)
m.add_line(chip_dx-(500+100)*2., 0)
m.add_taper(300, 0, 500, 100)
m.add_line(400, 0)
m.add_end(100, 0)
p += m
dx_f, dy_f = m.get_size()


# Resonators
res_lengths = [5500, 6500, 7500, 8455]
distance_between_res = 750.
coupling_length = 250
turn_radius = 50
gap_distance = 500
nb_turn = 15
spacing = 30
end_line = 100
ends = ['shunt', 'open_circular', 'open_fresnel', 'open']

# Create design polygon which will contain one feedline and one resonator
pp = PolygonSet()
for i, (res_length, end) in enumerate(zip(res_lengths, ends)):
    m = CPW(width=4, gap=2, layer=0)
    m.add_line(coupling_length, 0)
    m.add_turn(turn_radius, 'lb')
    m.add_line(0, -gap_distance)
    m.add_serpentine(total_length=res_length-coupling_length-gap_distance-turn_radius*np.pi/2.-end_line,
                    nb_turn=nb_turn,
                    spacing=spacing,
                    orientation='bottom',
                    starting='left',
                    )
    m.add_line(0, -end_line)
    if end=='shunt':
        pass
    elif end=='open_circular':
        m.add_circular_end('bottom')
    elif end=='open_fresnel':
        m.add_fresnel_end('bottom')
    elif end=='open':
        m.add_end(0, -m.gap)
    print('Resonator ', i, ' length: ', m.total_length, ' um')
    pp += m.translate(i*distance_between_res, -10.)

dx_r, dy_r = pp.get_size()
p += pp.translate(dx_f/2.-dx_r/2., 0.)

# Copy the feedline and the resonator 4 times
p += copy.copy(p).translate(0, -1500)
p += copy.copy(p).translate(0, -3000)


# Add label on the top right of the chip
t = Text('R0C0', 250)
dx_t, dy_t = t.get_size()
tot += t.translate(-dx_t+chip_dx-250, -dy_t-250)


# Center the design to the chip and the chip to the (0,0) of the cell
tot.center()
tot += p.center()

# Here, we copy the all chip and inverse its polarity
# This will create a polygon of where all the metal is
g = copy.deepcopy(tot).inverse_polarity()
# That polygon is is then cover by a grid of square by using the grid_cover
# function
g = grid_cover(g)

# Add polygons to cell
cell.add(tot)
cell.add(g)

# Save GDS file
lib.write_gds('cpw_resonator_with_grid_gnd.gds')
