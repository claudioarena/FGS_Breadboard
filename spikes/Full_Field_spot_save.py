import numpy as np
from FGS_Breadboard import fgs
import matplotlib.pyplot as plt
from scipy import signal
import time

exp_time = 8.0
gain = 1
gain_boost = False

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482", exp_time=exp_time, gain=gain, gain_boost=gain_boost)
image = fgs_var.get_frame(bias_correct=True)
# plt.imshow(image)
centroid = fgs_var.find_centroid_blind(5, 20, 1.5)
x_off, y_off = fgs_var.setup_AOI(40, 40, 1.5)
image = fgs_var.get_frame(bias_correct=True)
plt.imshow(image)

## Play with the following if you need high FPS
fgs_var.fgs_cam.set_exposure(8.0)
#fgs_var.fgs_cam.set_clock(86)
#print(fgs_var.fgs_cam.get_clock())
#max_fps = fgs_var.fgs_cam.get_framerate_limits()[1]
#fgs_var.fgs_cam.set_framerate(max_fps)
#print(1 / fgs_var.fgs_cam.get_framerate())
#fgs_var.fgs_cam.set_exposure(8.0)

x_grid = [100, 370, 640, 910, 1180]
y_grid = [100, 306, 512, 718, 924]


for x in x_grid:
	for y in y_grid:
		pos, errs, time, total_light, slit_light = fgs_var.pid_start_const_target(x, y, repetitions=3000,
		                                                                          timeout=20,
		                                                                          output_active=True,
		                                                                          subFrame_size=40, spot_frame_size=1.5)
		fgs_var.reset_AOI()
		fgs_var.get_bias_level()
		image = fgs_var.get_frame(bias_correct=True)
		# plt.imshow(image)
		name = "data/march2020/full_field_20um_CFL/field"
		name = name + "_" + str(x) + "_" + str(y) + ".png"
		plt.imsave(name, image, cmap='Greys_r', vmin = 0, vmax = 255)

fgs_var.close()