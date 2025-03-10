import numpy as np
from copy import copy,deepcopy

from pygdsdesign import (
    GdsLibrary,
    PolygonSet,
    MicroStripPolar,
    CPWPolar,
)


# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')


# Create empty polygon to which other polygon will be added
tot = PolygonSet()

m=MicroStripPolar(2, 0)
m.add_line(6)

# We use a taper to change the width of the line
m.add_taper(5, 4)
m.add_line(6)

# Same as above but with a smooth curve for the taper
m.add_taper_arctan(5, 6, smoothness=3)
m.add_line(6)

tot += m


m = CPWPolar(2, 4, 0)
m.add_line(10)

# We use a taper to change the width of the line
m.add_taper(5, 4, 3)
m.add_line(10)

# Same as above but with a smooth curve for the taper
m.add_taper_arctan(2, 6, 1, smoothness=3)
m.add_line(10)

tot += m.translate(0, 25)

cell.add(tot)
# Save GDS file
lib.write_gds('tapering.gds')