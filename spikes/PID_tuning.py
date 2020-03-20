import numpy as np
from FGS_Breadboard import fgs
import matplotlib.pyplot as plt
from scipy import signal

exp_time = 8.0
gain = 1
gain_boost = False

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482", exp_time=exp_time, gain=gain, gain_boost=gain_boost)
image = fgs_var.get_frame(bias_correct=True)
#plt.imshow(image)
centroid = fgs_var.find_centroid_blind(5, 40, 1.5)
x_off, y_off = fgs_var.setup_AOI(20, 20, 1.5)
image = fgs_var.get_frame(bias_correct=True)
plt.imshow(image)

## Play with the following if you need high FPS
fgs_var.fgs_cam.set_exposure(5.0)
fgs_var.fgs_cam.set_clock(86)
print(fgs_var.fgs_cam.get_clock())
max_fps = fgs_var.fgs_cam.get_framerate_limits()[1]
fgs_var.fgs_cam.set_framerate(max_fps)
print(1/fgs_var.fgs_cam.get_framerate())
fgs_var.fgs_cam.set_exposure(1.0)


##Try a step impulse
def step_x(time):
	if time < 2:
		return 637
	else:
		return 647


def step_y(time):
	if time < 3:
		return 534
	else:
		return 544


fgs_var.pid_P = 0.6
fgs_var.pid_D = 0.0
fgs_var.pid_I = 0.0
pos, errs, time, total_light, slit_light = fgs_var.pid_start(step_x, step_y, repetitions=1000, timeout=20, output_active=True,
                                                             subFrame_size=20, spot_frame_size=1.5)

print("RMS: {}".format(np.std(pos[-100:-1, 0])))
plt.figure()
plt.scatter(time, pos[:, 0], c='coral')
plt.figure()
plt.scatter(time, pos[:, 1], c='lightblue')
#plt.figure()
#plt.scatter(time, total_light, c='lightblue')
#plt.figure()
#plt.scatter(time, slit_light, c='lightblue')
plt.show()

fgs_var.close()