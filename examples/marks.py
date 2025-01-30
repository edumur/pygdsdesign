import time

import numpy as np

from pygdsdesign import GdsLibrary
from pygdsdesign.shapes import cross, crosses, global_marks_ebeam, chip_marks_ebeam, chip_marks_laser, dicing_saw_mark, mark_uv

start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

# cross is a generic function to generate a simple cross, by default it is centered
tot = cross(layer=1,
            datatype=1,
            width=10,
            h_length=50,
            v_length=50
            )

# crosses returns a simple cross at each position specified in the coordinates list
tot += crosses(coordinates=[(-200, -200), (-200, 200), (200, 200), (200, -200)])

# global ebeam marks generates a cross with additional features if wanted such as structures that help to find the center (directional structure) and a clear center point (squared_center)
tot+=global_marks_ebeam(squared_center=True).translate(-500, -500)

# chip marks ebeam generate a set of four crosses we use generally with our e-beam system
tot+=chip_marks_ebeam().translate(-500, 500)

# chip marks laser generate a set of four crosses we use generally with our laser system (the location is the center of the two squares that touch)
tot+=chip_marks_laser().translate(500, 500)

# alignment marks for UV lithography
tot+=mark_uv(layer_uv_mask=10,
             layer_metallisation=11
             ).translate(500, -500)

# handy dicing saw marks that will be entierly removed by the dicing process (size correspondsto the blade width, which depens on the material)
tot+=dicing_saw_mark().translate(-1500, -1500)
tot+=dicing_saw_mark().translate(-1500, 1500)
tot+=dicing_saw_mark(substrate='sapphire').translate(1500, -1500)
tot+=dicing_saw_mark(substrate='sapphire').translate(1500, 1500)

cell.add(tot)

lib.export_gds('marks.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')