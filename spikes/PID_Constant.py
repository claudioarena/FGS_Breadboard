import numpy as np
from FGS_Breadboard import fgs
import matplotlib.pyplot as plt
from scipy import signal

exp_time = 6.0
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
fgs_var.fgs_cam.set_exposure(6.0)
fgs_var.fgs_cam.set_clock(86)
print(fgs_var.fgs_cam.get_clock())
max_fps = fgs_var.fgs_cam.get_framerate_limits()[1]
fgs_var.fgs_cam.set_framerate(max_fps)
print(1/fgs_var.fgs_cam.get_framerate())
fgs_var.fgs_cam.set_exposure(6.0)

pos, errs, time, total_light, slit_light = fgs_var.pid_start_const_target(637, 534, repetitions=3000000, timeout=30,
                                                                          output_active=True,
                                                             subFrame_size=40, spot_frame_size=1.5)

plt.figure()
plt.scatter(time, pos[:, 0], c='coral')
plt.figure()
plt.scatter(time, pos[:, 1], c='lightblue')
#plt.figure()
#plt.scatter(time, total_light, c='lightblue')
#plt.figure()
#plt.scatter(time, slit_light, c='lightblue')
#plt.show()

fs = 1/ ( (time[500] - time[100]) / 400 )
##diversion from linear time: +- 0.002 sec
plt.figure()
f, Pxx_den = signal.periodogram(np.sqrt(np.square(pos[:,0])+np.square(pos[:,1])), fs)
plt.loglog(f, Pxx_den)
#plt.ylim([1e-7, 1e2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

fgs_var.close()