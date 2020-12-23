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
centroid = fgs_var.find_centroid_blind(5, 20, 1.5)
x_off, y_off = fgs_var.setup_AOI(40, 40, 1.5)
image = fgs_var.get_frame(bias_correct=True)
plt.imshow(image)

## Play with the following if you need high FPS
fgs_var.fgs_cam.set_exposure(8.0)
fgs_var.fgs_cam.set_clock(86)
print(fgs_var.fgs_cam.get_clock())
max_fps = fgs_var.fgs_cam.get_framerate_limits()[1]
fgs_var.fgs_cam.set_framerate(max_fps)
print(1/fgs_var.fgs_cam.get_framerate())
fgs_var.fgs_cam.set_exposure(8.0)


##Try a step impulse
def fun_x(time):
	#f = 1 / 10
	#return np.sin(2*3.14*f*time)*3 + 650
	if time < 10:
		return 670.0
	else:
		return 680.0


def fun_y(time):
	#f = 1 / 10
	#return np.sin(2*3.14*f*time + 3.14/2)*3 + 530
	#if time < 8:
#		return 530
#	else:
#		return 550
	return 530.0


pos, errs, time, total_light, slit_light = fgs_var.pid_start(fun_x, fun_y, repetitions=3000, timeout=20, output_active=True,
                                                             subFrame_size=40, spot_frame_size=1.5)

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