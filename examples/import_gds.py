import time

from pygdsdesign import GdsLibrary, PolygonSet, lateqs_logo

start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('TOP')


# Create empty polygon to which other polygon will be added
tot = PolygonSet()
tot += lateqs_logo(layer=2, datatype=1, name='logo')


# Add polygons to cell
cell.add(tot)

# Save GDS file
lib.export_gds('import_gds.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')
