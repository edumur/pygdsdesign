import numpy as np
import copy

from pygdsdesign import GdsLibrary, PolygonSet, Rectangle, Text, colors

offset_value = 0.250

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')

# Create empty polygon to which other polygon will be added
width  = 50
height = 10
gap_v  = 5
gap_h  = 10

tot = PolygonSet([[(0, 0)]])
r1  = Rectangle((0, 0),
                (width, height))

for i, (name, color) in enumerate(colors.items()):

    # Others is not a material
    if name=='others':
        continue

    # Add a colored rectangle
    r2 = copy.copy(r1)
    r2.names = [name]
    r2.layers = [i]
    tot += r2.translate(0, tot.get_size()[1]+gap_v)

    # Add a name before the colored rectangle
    if name=='':
        name='default'
    t = Text(text=name, size=100, layer=i)
    t.scale(height/t.get_size()[1])
    tot += t.translate(-t.get_size()[0] - gap_h, -t.get_bounding_box()[0][1] + r2.get_bounding_box()[0][1])


# Add polygons to cell
cell.add(tot)

# Save GDS file
lib.export_gds('layers.gds', fill=False)
