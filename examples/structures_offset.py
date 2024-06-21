import numpy as np
import copy

from pygdsdesign import GdsLibrary, PolygonSet, offset

layer_big   = {'layer': 1, 'name' : 'Al'}
layer_small = {'layer': 2, 'name' : 'Al', 'datatype' : 1}
offset_value = 0.250

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')

# Create empty polygon to which other polygon will be added
tot = PolygonSet()

# Will contains full size design
full = PolygonSet(**layer_big)

# Will contains offset size design
o = PolygonSet(**layer_small)

# First contact line
p1 = PolygonSet(**layer_big)
p1 > (-13, 0)
p1 > (0, 10)
p1 > (4, 0)
p1 > (0, 20)
p1 > (2, 0)
p1 > (0, 30)
p1 > (1, 0)
p1 > (0, -30)
p1 > (2, 0)
p1 > (0, -20)
p1 > (4, 0)
p1 > (0, -10)

dx, dy = p1.get_size()
p1.translate(dx/2, -dy-1)

full += p1
full += copy.copy(p1).rotate(np.deg2rad(90))
full += copy.copy(p1).rotate(np.deg2rad(180))
full += copy.copy(p1).rotate(np.deg2rad(270))

o = offset(full, -offset_value, join_first=True, **layer_small)


tot += full
tot += o

# Add polygons to cell
cell.add(tot.merge().center())

# Save GDS file
lib.export_gds('structures_offset.gds')
