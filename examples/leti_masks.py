import time

import numpy as np

from pygdsdesign import GdsLibrary
from pygdsdesign.shapes import qubit_layer_19, qubit_layer_42

start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

tot = qubit_layer_42(layer=52)
tot+= qubit_layer_19(layer=51)

cell.add(tot)

lib.export_gds('leti_masks.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')