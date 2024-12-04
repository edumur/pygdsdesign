from pygdsdesign import (
    GdsLibrary,
    PolygonSet,
    Rectangle,
)
from pygdsdesign.shapes import lateqs_logo
from pygdsdesign.operation import (
    offset,
    grid_cover,
    subtraction,
)

# define layer
layer_background = {
    "layer": 1,
    "name": "background",
    "datatype": 0,
    "color": "#ff00ff",
}

layer_grid = {
    "layer": 2,
    "name": "grid",
    "datatype": 0,
    "color": "#13c24b",
}


# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell("TOP")

# Create empty polygon to which other polygon will be added
tot = PolygonSet()

# the polygon we will operate on
poly = lateqs_logo()

# Make a rectangle big enough for our purposes
r = offset(
    Rectangle(*lateqs_logo().get_bounding_box()),
    20,
)

# make the negative of the group logo
r = subtraction(r, poly, **layer_background)

# Use that negative for the grid cover operation
grid = grid_cover(
    polygons=r,
    square_width=12,
    square_gap=23,
    safety_margin=10,
    centered=True,
    noise=12,
    only_square=False,
    **layer_grid
)



tot += r + grid

# Add polygons to cell
cell.add(tot)

lib.export_gds("gridcover.gds")