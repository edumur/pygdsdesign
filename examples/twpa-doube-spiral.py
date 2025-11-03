import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, minimize_scalar
from math import sqrt, erf, atan2
from scipy.integrate import quad, cumulative_trapezoid

from scipy.interpolate import InterpolatedUnivariateSpline
Coordinate = tuple[float, float]

import time
from pygdsdesign import (
    GdsLibrary,
    Text,
    MicroStrip,
    MicroStripPolar,
    PolygonSet,
    Rectangle,
)
# from pygdsdesign.transmission_lines import (
#     CPW,
#     MicroStripPolar,
# )
from pygdsdesign.operation import(
    replace_by_photonic_crystal,
    inverse_polarity,
    crop,
    merge,
)


chip_dx = 9800
chip_dy = 9800
chip_border_margin = 200

microstrip_width = 5

bonding_pad_width = 300
bonding_pad_length = 375
bonding_pad_taper = 250


twpa_length = 30000
twpa_nb_turn=7
twpa_inner_radius = 1000
twpa_nb_points_per_turn=5000
twpa_link_nb_points=1001

# photonic crystal
# may contains multiple harmonics
# twpa_period_lengths = [50, 105]
# twpa_w_mins = [5, 10]
# twpa_w_maxs = [10, 20]
# twpa_phases = [np.pi/2, np.pi/2]
# twpa_period_cos_powers = [1, 1]
twpa_period_lengths = [50, ]
twpa_w_mins = [5, ]
twpa_w_maxs = [10, ]
twpa_phases = [np.pi, ]
twpa_period_cos_powers = [1, ]


layer_nbn = {
    "layer": 1,
    "name": "nbn",
    "datatype": 0,
    "color": "#ff00ff",
}


layer_chip = {
    "layer": 10,
    "name": "chip",
    "datatype": 0,
    "color": "#f5552f",
}



start = time.time()

# The GDSII file is called a library, which contains multiple cells.
lib = GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

tot = PolygonSet()




# ######## Bonding pad
m = MicroStrip(width=bonding_pad_width, **layer_nbn)
m.add_line(bonding_pad_length, 0)
m.add_taper_cosec(bonding_pad_taper, 0, microstrip_width, 25)
m.add_line(1, 0)
m = merge(m)
m.fillet(50)
m = crop(m, 'right', 1, **layer_nbn)
# correct crop slight error
n = m.polygons[0][:,0].argmax()
m.polygons[0][:,1][n]   = bonding_pad_width/2 - microstrip_width/2
m.polygons[0][:,1][n+1] = bonding_pad_width/2 + microstrip_width/2

# left
m.translate(chip_border_margin, -chip_dy/2)
tot += m
# save coordinates for later
((x0, y0), (x1, y1)) = m.get_bounding_box()
x_bp_left = x1
y_bp_left = (y0+y1)/2

# right
m = deepcopy(m).flip('y').translate(chip_dx-2*chip_border_margin-m.get_size()[0], 0)
tot += m
# save coordinates for later
((x0, y0), (x1, y1)) = m.get_bounding_box()
x_bp_right = x0
y_bp_right = (y0+y1)/2




line_spacing = 200
m = MicroStripPolar(width=5, angle=0, **layer_nbn)

theta = np.linspace(0, 2*np.pi*twpa_nb_turn, twpa_nb_turn*twpa_nb_points_per_turn)

# Spiral 1: from outer left → center
r1 = twpa_inner_radius + line_spacing*2* theta / (2*np.pi)
x1 = -r1 * np.cos(theta)
y1 = r1 * np.sin(theta)

dx1 = np.gradient(theta, x1)
dy1 = np.gradient(theta, y1)

f_dx1 = InterpolatedUnivariateSpline(theta, x1, k=1)
f_dy1 = InterpolatedUnivariateSpline(theta, y1, k=1)

spiral1_final_angle = np.angle(dx1+1j*dy1)[0]-np.pi/2

spiral1_length = quad(m.curve_length, theta[0], theta[-1], args=(f_dx1, f_dy1))[0]

# Spiral 2: from outer right → center
r2 = twpa_inner_radius + line_spacing*2* theta / (2*np.pi)
x2 = r2 * np.cos(theta)
y2 = -r2 * np.sin(theta)

dx2 = np.gradient(theta, x2)
dy2 = np.gradient(theta, y2)

f_dx2 = InterpolatedUnivariateSpline(theta, x2, k=1)
f_dy2 = InterpolatedUnivariateSpline(theta, y2, k=1)

spiral2_final_angle = np.angle(dx2+1j*dy2)[0]-np.pi/2

spiral2_length = quad(m.curve_length, theta[0], theta[-1], args=(f_dx2, f_dy2))[0]


# Connect the centers of both spirals
start_pt = np.array([x1[0], y1[0]])  # innermost of spiral 1
end_pt = np.array([x2[0], y2[0]])    # innermost of spiral 2

def link(length, sigma, delta_angle):

    # Gaussian curve to link
    _, _, x_link_1, y_link_1 = m._get_gaussian_curve(
        delta_angle=delta_angle,
        length=length,
        nb_points=twpa_link_nb_points,
        initial_angle=-spiral1_final_angle,
        origin=start_pt,
        sigma=sigma,
    )

    # Gaussian curve to link
    _, _, x_link_2, y_link_2 = m._get_gaussian_curve(
        delta_angle=delta_angle,
        length=length,
        nb_points=twpa_link_nb_points,
        initial_angle=-spiral2_final_angle,
        origin=end_pt,
        sigma=sigma,
    )
    return x_link_1, y_link_1, x_link_2, y_link_2

def func(p):
    x_link_1, y_link_1, x_link_2, y_link_2 = link(*p)

    return (x_link_1[-1]-x_link_2[-1])**2 + (y_link_1[-1] - y_link_2[-1])**2

# get the optimal gaussian turn connecting the two spiral and joing perfectly in between
# length, sigma, delta_angle
p0 = [7.38323322e+02, 7.52601024e+02, 2.88894593e+00]
r = minimize(func, p0)
p0 = r.x

# Get the link coordinates
x_link_1, y_link_1, x_link_2, y_link_2 = link(*p0)

# Actual TWPA length
total_length = spiral1_length + p0[0]*2 + spiral2_length
print('twpa total length: {:.3f}cm'.format(total_length/1e4))

# Get the twpa coordinates
# here, the points along the curve (called arc length) are not equidistant
x_tot = np.concatenate((x1[::-1][:-1], x_link_1[:-1], x_link_2[::-1][:-1], x2))
y_tot = np.concatenate((y1[::-1][:-1], y_link_1[:-1], y_link_2[::-1][:-1], y2))

# resample it at equal arc-length intervals.
# 1. Compute the cumulative distance (arc length) along the curve
dx = np.diff(x_tot)
dy = np.diff(y_tot)
dist = np.sqrt(dx**2 + dy**2)
# cumulative distance array
s = np.concatenate(([0], np.cumsum(dist)))
# for example, how many equidistant points you want
s_uniform = np.linspace(0, total_length, len(x_tot))

# 3. Interpolate x and y as functions of the arc length
x_interp = InterpolatedUnivariateSpline(s, x_tot)
y_interp = InterpolatedUnivariateSpline(s, y_tot)

# 4. Compute the new coordinates at the uniform positions
x_tot = x_interp(s_uniform)
y_tot = y_interp(s_uniform)


# get derivative for the angle for the photonic crystal
dx_tot = np.gradient(x_tot, s_uniform)
dy_tot = np.gradient(y_tot, s_uniform)
theta = np.angle(dx_tot+1j*dy_tot)


def fun_pc(x: np.ndarray) -> np.ndarray:

    tot = 0
    for (period_length,
         w_min,
         w_max,
         phase,
         period_cos_power) in zip(twpa_period_lengths,
                                  twpa_w_mins,
                                  twpa_w_maxs,
                                  twpa_phases,
                                  twpa_period_cos_powers):
        y = (np.cos(x*2*np.pi/period_length + phase) + 1)
        y = y**period_cos_power
        y = y/y.max()
        y = y*(w_max - w_min)
        y = y + w_min

        tot += y

    return tot

# Compute the boundaries of the strip
x_lower = x_tot + np.cos(theta - np.pi/2) * fun_pc(s_uniform)
y_lower = y_tot + np.sin(theta - np.pi/2) * fun_pc(s_uniform)

x_upper = x_tot - np.cos(theta - np.pi/2) * fun_pc(s_uniform)
y_upper = y_tot - np.sin(theta - np.pi/2) * fun_pc(s_uniform)

# Concatenate upper and lower boundaries to form polygon
x_strip = np.concatenate((x_lower, x_upper[::-1]))
y_strip = np.concatenate((y_lower, y_upper[::-1]))

twpa = PolygonSet(
    polygons=[np.vstack((x_strip, y_strip)).T],
    **layer_nbn
).translate(chip_dx/2, -chip_dy/2)


###### link between left pad and twpa

# find info of the beginning of the twpa
x = np.array([twpa.polygons[0][:,0][0], twpa.polygons[0][:,0][-1]])
y = np.array([twpa.polygons[0][:,1][0], twpa.polygons[0][:,1][-1]])
w = np.sqrt((twpa.polygons[0][:,0][-1]-twpa.polygons[0][:,0][0])**2 + (twpa.polygons[0][:,1][-1]-twpa.polygons[0][:,1][0])**2)
s = np.linspace(0, 1, 2)
dx = np.gradient(x, s)
dy = np.gradient(y, s)
theta = np.angle(dx + 1j*dy)

# find best length and angle
def link_bp_left_twpa(length_curve, angle, length_straight1, length_straight2):
    m = MicroStripPolar(
        width=w,
        angle=theta[0]+np.pi/2,
        ref=((twpa.polygons[0][:,0][-1]+twpa.polygons[0][:,0][0])/2,
            (twpa.polygons[0][:,1][-1]+twpa.polygons[0][:,1][0])/2),
        **layer_nbn)
    m.add_line(length_straight1)
    m.add_gaussian_turn(length_curve, angle)
    m.add_taper_arctan(length_straight2, microstrip_width)

    return m

def func(p):
    m = link_bp_left_twpa(*p)
    res = (m.ref[0] - x_bp_left)**2 \
        + (m.ref[1] - y_bp_left)**2 \
        + (m._angle - np.pi)**2
    return res

# p = [150, theta[0]+np.pi/2, 150, 150]
p = [131.4746306, 1.55330966, 74.50268882, 197.0038111]
r = minimize(func, p)
p = r.x
# print(p)
m = link_bp_left_twpa(*p)

# make link perfect
temp = m.polygons[-1]
n = int(len(temp[:,0])/2)
m.polygons[-1][n][0] = x_bp_left
m.polygons[-1][n-1][0] = x_bp_left
m.polygons[-1][n][1] = y_bp_left + microstrip_width/2
m.polygons[-1][n-1][1] = y_bp_left - microstrip_width/2


tot += m






###### link between right pad and twpa

# find info of the beginning of the twpa
n = int(len(twpa.polygons[0][:,0])/2)
x = np.array([twpa.polygons[0][:,0][n], twpa.polygons[0][:,0][n-1]])
y = np.array([twpa.polygons[0][:,1][n], twpa.polygons[0][:,1][n-1]])
w = np.sqrt((twpa.polygons[0][:,0][n-1]-twpa.polygons[0][:,0][n])**2 + (twpa.polygons[0][:,1][n-1]-twpa.polygons[0][:,1][n])**2)
s = np.linspace(0, 1, 2)
dx = np.gradient(x, s)
dy = np.gradient(y, s)
theta = np.angle(dx + 1j*dy)

# find best length and angle
def link_bp_right_twpa(length_straight1, length_curve1, angle1, length_curve2, angle2, length_straight3):
    m = MicroStripPolar(
        width=w,
        angle=theta[0]+np.pi/2,
        ref=((twpa.polygons[0][:,0][n-1]+twpa.polygons[0][:,0][n])/2,
            (twpa.polygons[0][:,1][n-1]+twpa.polygons[0][:,1][n])/2),
        **layer_nbn)
    m.add_line(length_straight1)
    # m.add_line(length_straight1)
    m.add_gaussian_turn(length_curve1, -angle1)
    m.add_gaussian_turn(length_curve2, -angle2)
    m.add_gaussian_turn(length_curve2, angle2)
    m.add_taper_arctan(length_straight3, microstrip_width)

    return m

def func(p):
    m = link_bp_right_twpa(*p)
    res = (m.ref[0] - x_bp_right)**2 \
        + (m.ref[1] - y_bp_right)**2 \
        + (m._angle)**2
    return res

p = [10, 140, theta[0]+np.pi/2, 130, theta[0]+np.pi/2, 40]
r = minimize(func, p)
p = r.x
# print(p)
m = link_bp_right_twpa(*p)

# make link perfect
temp = m.polygons[-1]
n = int(len(temp[:,0])/2)
m.polygons[-1][n][0] = x_bp_right
m.polygons[-1][n-1][0] = x_bp_right
m.polygons[-1][n][1] = y_bp_right - microstrip_width/2
m.polygons[-1][n-1][1] = y_bp_right + microstrip_width/2

tot += m



tot += twpa

tot += Rectangle((0,0), (chip_dx, -chip_dy), **layer_chip)

cell.add(tot)



lib.export_gds('twpa-doube-spiral.gds')

stop = time.time()

print('Generation time:', stop-start, ' s')
