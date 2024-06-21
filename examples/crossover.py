import time

from pygdsdesign import GdsLibrary, PolygonSet, shapes

# Layer definitions (0 is reserved for default)
layers_crossover = {'layer_dielectric' : 2,
                    'name_dielectric' : 'SiO2',
                    'layer_metal' : 3,
                    'name_metal' : 'Al'}
layers_daisychain = {'layer_NbN_etch' : 1,
                     'name_NbN_etch' : 'NbN',
                     'layer_dielectric' : 2,
                     'name_dielectric' : 'SiO2',
                     'layer_metal' : 3,
                     'name_metal' : 'Al'}

# some dimensions
gap = 50
pad_size = 200

# Start the design

start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

# Create empty polygon to which other polygon will be added
tot = PolygonSet()

for i in range(20):
    tot += shapes.crossover(**layers_crossover).translate(150*i, 0)

tot.center()

# Add some daisychains to test
# a geometry for briding our feedlines
tot += shapes.daisychain(num=1, l=160, w=5, **layers_daisychain).translate(0, 3000)
tot += shapes.daisychain(num=2, l=20, w=50, **layers_daisychain).translate(0, 2500)
tot += shapes.daisychain(num=4, l=10, w=1, **layers_daisychain).translate(0, 2000)
tot += shapes.daisychain(num=8, l=5, w=5, **layers_daisychain).translate(0, 1500)
tot += shapes.daisychain(num=16, l=100, w=1, **layers_daisychain).translate(0, 1000)
# add one with undercut for the dielectric layer on different datatype
tot += shapes.daisychain(num=16, l=100, w=1, u=0.5, **layers_daisychain).translate(0, 1000)


# Add polygons to cell
cell.add(tot)

lib.export_gds('crossover.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')