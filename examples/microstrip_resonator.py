import copy

from pygdsdesign import GdsLibrary, PolygonSet, Rectangle, Text, MicroStrip

# Example of a microstrip transmission line where a multiple of identical
# resonator are coupled to in order to create a stop band


# Chip size of 10mm x 7mm10
chip_dx = 10000
chip_dy = 10000

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


# Default resonator
r = MicroStrip(width=2, layer=0)
r.add_line(150, 0)
r.add_fresnel_turn(25, 'lb')
r.add_fresnel_turn(25, 'tl')
r.add_line(-75, 0)
r.add_fresnel_turn(10, 'rb')
r.add_serpentine(total_length=1500,
                 nb_turn=10,
                 spacing=20,
                 orientation='bottom',
                 starting='left',
                 turn='fresnel',
                 )

# Feedline
pad_width = 500
pad_length = 500
offset_length = 500
taper_length = 300
turn_radius = 50
nb_turn = 8
spacing = 700
end_line = 100

m = MicroStrip(width=pad_width, layer=0)
m.add_line(pad_length, 0)
m.add_taper(taper_length, 0, 4)
m.add_line(offset_length, 0)
m.add_serpentine(total_length=50000,
                 nb_turn=nb_turn,
                 spacing=spacing,
                 orientation='right',
                 starting='left',
                 )
m.add_line(offset_length, 0)
m.add_taper(taper_length, 0, pad_width)
m.add_line(pad_length, 0)
m.add_and_follow(structure=r,
                 pitch=1000,
                 start_offset=pad_length+offset_length,
                 stop_offset=pad_length+offset_length,
                 noise=100,
                 distance=144.5)

tot += m.center().translate(chip_dx/2, -chip_dy/2)

# Chip label
t = Text('R0C0', 250)
dx_t, dy_t = t.get_size()
tot += t.translate(-dx_t+chip_dx-250, -dy_t-250)


# Center the design to the chip and the chip to the (0,0) of the cell
tot.center()
tot += p.center()

# Add polygons to cell
cell.add(tot)

# Save GDS file
lib.write_gds('microstrip_resonator.gds')
