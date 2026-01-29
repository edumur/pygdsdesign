import numpy as np
from copy import copy,deepcopy

from pygdsdesign import (
    GdsLibrary,
    PolygonSet,
    Rectangle,
)

from pygdsdesign.operation import snap


# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')

# Create empty polygon to which other polygon will be added
tot = PolygonSet()


# Create two rectangles that we want to snap
p1 = Rectangle((-20, -10), (0, 0)).rotate(-0.15).flip('y')
p2 = Rectangle((0, 0), (30, 5)).translate(23, 23).rotate(-0.23)

# Show the original polygons
tot += deepcopy(p1) + deepcopy(p2)

# Decide wich sides we want to snap
port1 = 1
port2 = 2

# Mean 1 : via operation (doesn't change the original polygons)
# snap them and translate them to show the result
# print(snap(p1, port1, p2, port2).polygons)
tot += snap(p1, port1, p2, port2).translate(0, 50)
tot += snap(p1, port1, p2, port2, show_port=True).translate(50, 50)


# Mean 2 : via polygons methods (change the original polygons)
# snap them and translate them to show the result
# print(snap(p1, port1, p2, port2).polygons)
p3 = deepcopy(p1).change_layer(layer=10)
tot += p3.snap_to(port1, p2, port2, show_port=True)

cell.add(tot)
# Save GDS file
lib.write_gds('snap.gds')