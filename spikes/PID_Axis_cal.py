import numpy as np
from FGS_Breadboard import fgs
import matplotlib.pyplot as plt
from scipy import signal

exp_time = 9.0
gain = 1
gain_boost = False

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482", exp_time=exp_time, gain=gain, gain_boost=gain_boost)
image = fgs_var.get_frame(bias_correct=True)
#plt.imshow(image)
centroid = fgs_var.find_centroid_blind(5, 20, 1.5)
x_off, y_off = fgs_var.setup_AOI(40, 40, 1.5)
image = fgs_var.get_frame(bias_correct=True)
plt.imshow(image)

## Play with the following if you need high FPS
fgs_var.fgs_cam.set_exposure(exp_time)
#fgs_var.fgs_cam.set_clock(86)
#print(fgs_var.fgs_cam.get_clock())
#max_fps = fgs_var.fgs_cam.get_framerate_limits()[1]
#fgs_var.fgs_cam.set_framerate(max_fps)
#print(1/fgs_var.fgs_cam.get_framerate())
#fgs_var.fgs_cam.set_exposure(exp_time)

n_average = 10
v_start = 20
v_end = 80
v_n = 1
pause_time = 10
cx_X, cy_X, v_range_X, d_X = fgs_var._calibrate_axis(fgs_var.X_AXIS, n_average, v_start, v_end, v_n, pause_time)
cx_Y, cy_Y, v_range_Y, d_Y = fgs_var._calibrate_axis(fgs_var.Y_AXIS, n_average, v_start, v_end, v_n, pause_time)
cx_Z, cy_Z, v_range_Z, d_Z = fgs_var._calibrate_axis(fgs_var.Z_AXIS, n_average, v_start, v_end, v_n, pause_time)

axis_sense = fgs_var.axis_sensitivity
print("Axis Sensitivity: {}".format(fgs_var.axis_sensitivity))
axis_angle = fgs_var.axis_angle
print("Axis Angle: {}".format(fgs_var.axis_angle))
axis_x_proj = fgs_var.axis_x_proj
print("Axis X Proj: {}".format(fgs_var.axis_x_proj))
axis_y_proj = fgs_var.axis_y_proj
print("Axis Y Proj: {}".format(fgs_var.axis_y_proj))
