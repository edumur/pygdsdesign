import numpy as np
import time

from pygdsdesign import GdsLibrary, Text, CPW


start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

# Test base
start_base = time.time()
c = CPW(width=2, gap=3, layer=6)
c.add_line(0, 15)
c.add_line(-15, 0)
c.add_line(0, -15)

c.add_line(15, 0)
c.add_turn(15, 'lb')
c.add_line(0, -15)
c.add_turn(15, 'tl')
c.add_line(-15, 0)
c.add_line(-20, 0)
c.add_turn(15, 'rt')
c.add_line(0, 15)
c.add_turn(15, 'bl')
c.add_line(-15, 0)
c.add_turn(15, 'rb')
c.add_line(0, -75)
c.add_turn(15, 'tr')
c.add_line(10, 0)
c.add_turn(15, 'lt')
c.add_line(0, 15)
c.add_turn(15, 'br')
c.add_line(15, 0)
c.add_turn(15, 'lb')
c.add_line(0, -75)
c.add_turn(15, 'tr')

c.add_line(5, 0)
c.add_taper(20, 0, 10, 23)
c.add_line(5, 0)
c.add_taper(20, 0, 2, 5)
c.add_turn(15, 'lb')
c.add_line(0, -5)
c.add_taper(0, -20, 5, 11)
c.add_line(0, -5)
c.add_taper(0, -20, 9, 2)
c.add_line(0, -5)
c.add_turn(15, 'tl')
c.add_line(-5, 0)
c.add_taper(-20, 0, 2, 5)
c.add_line(-5, 0)
c.add_taper(-20, 0, 9, 17)
c.add_line(-5, 0)
c.add_turn(15, 'rt')
c.add_line(0, 5)
c.add_taper(0, 10, 2, 3)
c.add_turn(15, 'bl')
c.add_line(-500, 0)
c.add_turn(15, 'rt')
c.add_line(0, 500)
c.add_turn(15, 'br')
# c.add_end(3, 0)
c.add_turn(15, 'lt')
# c.add_end(0, 3)
c.add_turn(15, 'bl')
# c.add_end(-3, 0)
c.add_turn(15, 'rb')
c.add_end(0, -3)

# c.inverse_polarity()
bounding_polygon = c.bounding_polygon.change_layer(10)
c+=bounding_polygon
cell.add(c)

stop_base = time.time()
print('Test base: {:.3f}s'.format(stop_base-start_base))

# Test circular turn, should form a close loop
start_circular = time.time()
m = CPW(2, 3, layer=2, name='loop')
m.add_line(0, 10)
m.add_turn(50, 'br')
m.add_line(10, 0)

m.add_turn(50, 'lt')
m.add_line(0, 10)
m.add_turn(50, 'br')
m.add_line(10, 0)
m.add_turn(50, 'lb')
m.add_line(0, -10)
m.add_turn(50, 'tr')
m.add_line(10, 0)
m.add_turn(50, 'lb')
m.add_line(0, -10)
m.add_turn(50, 'tl')
m.add_line(-10, 0)
m.add_turn(50, 'rb')
m.add_line(0, -10)
m.add_turn(50, 'tl')
m.add_line(-10, 0)
m.add_turn(50, 'rt')
m.add_line(0, 10)
m.add_turn(50, 'bl')
m.add_line(-10, 0)
m.add_turn(50, 'rt')

m.center()
m += Text('Circular loop', 50, (0, 0), layer=2, name='loop').center().translate(0, 250)

bounding_polygon = m.bounding_polygon.change_layer(10)
m+=bounding_polygon
cell.add(m.translate(-600, 600))

stop_circular = time.time()
print('Test circular: {:.3f}s'.format(stop_circular-start_circular))






# Test serpentine Circular
start_serpentine = time.time()
m = CPW(2, 3, layer=3, name='serpentine')

m.add_line(250, 0)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='right')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='right')
m.add_line(0, 250)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='top')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='top')
m.add_line(-500, 0)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='left')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='left')
m.add_line(0, -500)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='bottom')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='bottom')

m.center()
m += Text('Test serpentine circular', 50, (0, 0), layer=3, name='serpentine').center().translate(0, 600)
bounding_polygon = m.bounding_polygon.change_layer(10)
m+=bounding_polygon
cell.add(m.translate(1500, 1500))

stop_serpentine = time.time()
print('Test serpentine circular: {:.3f}s'.format(stop_serpentine-start_serpentine))




# Test serpentine Fresnel
start_serpentine = time.time()
m = CPW(2, 3, layer=3, name='serpentine')

m.add_line(250, 0)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='right', turn='fresnel')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='right', turn='fresnel')
m.add_line(0, 250)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='top', turn='fresnel')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='top', turn='fresnel')
m.add_line(-500, 0)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='left', turn='fresnel')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='left', turn='fresnel')
m.add_line(0, -500)
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='right', orientation='bottom', turn='fresnel')
m.add_serpentine(total_length=1000, nb_turn=4, spacing=35, starting='left', orientation='bottom', turn='fresnel')

m.center()
m += Text('Test serpentine fresnel', 50, (0, 0), layer=3, name='serpentine').center().translate(0, 600)

bounding_polygon = m.bounding_polygon.change_layer(10)
m+=bounding_polygon
cell.add(m.translate(0, 1500))

stop_serpentine = time.time()
print('Test serpentine fresnel: {:.3f}s'.format(stop_serpentine-start_serpentine))



# Test spiral
start_spiral = time.time()
def spiral(u, args):

    r = args[1]*2.*args[2]*u + args[0]/2.
    theta = args[1]*2.*np.pi*u
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def dspiral_dt(u, args):

    r = args[1]*2.*args[2]*u
    theta = args[1]*2.*np.pi*u
    dx_dt = -args[1]*2.*np.pi*np.sin(theta)*r + np.cos(theta)*args[1]*2.*args[2]
    dy_dt =  args[1]*2.*np.pi*np.cos(theta)*r + np.sin(theta)*args[1]*2.*args[2]
    return dx_dt, dy_dt


t = np.linspace(0, 1, 10000)
inner_diameter = 10
nb_turn = 40
spiral_spacing = 10.
spiral_width = 2.

m = CPW(2, 3, layer=4, name='spiral')
m.add_parametric_curve(spiral, dspiral_dt, t,
                                args=(inner_diameter, nb_turn, spiral_spacing))
m.add_line(0, 100)

m.center()
m += Text('Test spiral', 50, (0, 0), layer=4, name='spiral').center().translate(0, 900)
bounding_polygon = m.bounding_polygon.change_layer(10)
m+=bounding_polygon
cell.add(m.translate(500, -1000))

stop_spiral = time.time()
print('Test spiral: {:.3f}s'.format(stop_spiral-start_spiral))





# Test Fresnel turn, should form a close loop
start_fresnel = time.time()
m = CPW(4, 5, layer=5, name='Fresnel')
m.add_line(0, 10)
m.add_turn(50, 'br')
m.add_line(10, 0)

m.add_fresnel_turn(50, 'lt', nb_points=51)
m.add_line(0, 10)
m.add_fresnel_turn(50, 'br')
m.add_line(10, 0)
m.add_fresnel_turn(50, 'lb')
m.add_line(0, -10)
m.add_fresnel_turn(50, 'tr')
m.add_line(10, 0)
m.add_fresnel_turn(50, 'lb')
m.add_line(0, -10)
m.add_fresnel_turn(50, 'tl')
m.add_line(-10, 0)
m.add_fresnel_turn(50, 'rb')
m.add_line(0, -10)
m.add_fresnel_turn(50, 'tl')
m.add_line(-10, 0)
m.add_fresnel_turn(50, 'rt')
m.add_line(0, 10)
m.add_fresnel_turn(50, 'bl')
m.add_line(-10, 0)
m.add_turn(50, 'rt')

m.center()
m += Text('Fresnel loop', 50, (0, 0), layer=5, name='Fresnel').center().translate(0, 250)
bounding_polygon = m.bounding_polygon.change_layer(10)
m+=bounding_polygon
cell.add(m.translate(500, 500))

stop_fresnel = time.time()
print('Test fresnel: {:.3f}s'.format(stop_fresnel-start_fresnel))


lib.export_gds('cpw.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')