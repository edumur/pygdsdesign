import numpy as np
import time

from pygdsdesign import GdsLibrary, Text, MicroStrip, inverse_polarity


start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')


# Test basic class
m = MicroStrip(width=2, layer=6, name='test1')
m.add_line(0, 8)
m.add_line(8, 0)
m.add_line(0, 10)
m.add_line(-16, 0)
m.add_line(0, -6)
m.add_line(4, 0)
m.add_line(0, -16)
m.add_line(20, 0)
m.add_turn(15, 'lt')
m.add_line(0, 5)

m.add_line(5, 0)
m.add_line(20, 0)
m.add_turn(15, 'lt')
m.add_turn(5, 'lb')
m.add_line(0, -5)

m.add_line(0, -10)
m.add_turn(5, 'tl')
m.add_line(-5, 0)

m.add_line(0, -10)
m.add_turn(5, 'tr')
m.add_line(5, 0)

m.add_line(0, -10)
m.add_line(-10, 0)
m.add_turn(5, 'rb')
m.add_line(0, -5)

m.add_line(0, -10)
m.add_line(-10, 0)
m.add_turn(5, 'rt')
m.add_line(0, 5)

m.add_line(0, 10)
m.add_turn(5, 'br')
m.add_line(5, 0)

m.add_line(0, 60)
m.add_turn(5, 'bl')
m.add_line(-5, 0)
m.add_line(-5, 0)

m.add_taper(-10, 0, 6)
m.add_line(-8, 0)
m.add_line(-8, 0)

m.add_turn(4, 'rt')
m.add_turn(4, 'br')
m.add_taper(10, 0, 6)
m.add_line(8, 0)
m.add_line(8, 0)
m.add_taper(10, 0, 2)
m.add_line(8, 0)
m.add_line(8, 0)

m.add_turn(4, 'lt')
m.add_line(0, 8)
m.add_line(0, 8)
m.add_taper(0, 4, 5)
m.add_line(0, 8)
m.add_line(0, 8)

m.add_turn(4, 'br')
m.add_turn(4, 'lb')
m.add_line(0, -8)
m.add_line(0, -8)
m.add_taper(0, -4, 10)
m.add_line(0, -8)
m.add_line(0, -8)
m.add_turn(10, 'tr')
m.add_line(10, 0)
m.add_taper_cosec(10, 0, 2)
m.add_line(5, 0)
m.add_taper_cosec(10, 0, 5)
m.add_line(5, 0)

m.add_turn(5, 'lt')
m.add_line(0, 5)
m.add_taper_cosec(0, 10, 1)
m.add_taper_cosec(0, 12, 6)
m.add_line(0, 5)
m.add_taper_cosec(0, 10, 20)
m.add_taper_cosec(0, 12, 2)
m.add_line(0, 11.98)
m.add_turn(5, 'bl')
m.add_line(-5, 0)
m.add_line(-5, 0)
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_turn(5, 'rt')
m.add_turn(5, 'bl')
m.add_line(-1, 0)
m.add_taper_cosec(-10, 0, 5.36)
m.add_taper_cosec(-13.56, 0, 11.98)
m.add_taper_cosec(-8, 0, 2)
m.add_line(-1, 0)
m.add_turn(5, 'rb')
m.add_line(0, -2)
m.add_taper_cosec(0, -5, 4)
m.add_taper_cosec(0, -6, 6.89)
m.add_taper_cosec(0, -3, 2, 0.5)
m.add_line(0, -2)

m.layer=2
m.add_line(0, -3)

m = inverse_polarity(m)
m.translate(100, -40)

m.rotate(25, center=m.get_center())

cell.add(m)




# Test circular turn, should form a close loop
m = MicroStrip(2, 0, name='test2')
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
m += Text('Circular loop', 50, (0, 0), name='test2').center().translate(0, 250)

cell.add(m.translate(-500, 500))






# Test serpentine
m = MicroStrip(2, 0, datatype=6, name='test3')

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
m += Text('Test serpentine circular', 50, (0, 0), datatype=6, name='test3').center().translate(0, 600)

cell.add(m.translate(1500, 1500))



# Test serpentine
m = MicroStrip(2, 0, datatype=6, name='test3')

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
m += Text('Test serpentine fresnel', 50, (0, 0), datatype=6, name='test3').center().translate(0, 600)

cell.add(m.translate(0, 1500))




# Test spiral
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
spiral_spacing = 3.
spiral_width = 2.

m = MicroStrip(width=spiral_width, layer=3, name='test4')
m.add_parametric_curve(spiral, dspiral_dt, t,
                                args=(inner_diameter, nb_turn, spiral_spacing))
m.add_line(0, 100)

m.center()
m += Text('Test spiral', 50, (0, 0), layer=3, name='test4').center().translate(0, 400)

cell.add(m.translate(0, -750))





# Test Fresnel turn, should form a close loop
m = MicroStrip(2, layer=6, name='test5')
m.add_line(0, 10)
m.add_turn(50, 'br')
m.add_line(10, 0)

m.add_fresnel_turn(50, 'lt')
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
m += Text('Fresnel loop', 50, (0, 0), layer=6, name='test5').center().translate(0, 250)

cell.add(m.translate(500, 500))



lib.export_gds('microstrip.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')
