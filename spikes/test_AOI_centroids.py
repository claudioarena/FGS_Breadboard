import uc480
from PIL import Image, ImageFilter
from scipy import ndimage
import numpy as np
from scipy.optimize import leastsq
from mdt69x import Controller
import time
from sklearn import metrics
import math
from matplotlib import pyplot as plt
from simple_pid import PID
from FGS_Breadboard import fgs
from FGS_Breadboard import Utilities
import time
import matplotlib.pyplot as plt
import pickle

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482")
fgs_var.setup_camera(13)
bias = fgs_var.get_bias_level()
image = fgs_var.get_frame() - bias
#plt.imshow(image)

blur_radius = 5
subframe_size = 10
l_subframe_size = subframe_size * 4
centroid = Utilities.centroid_approximate(image, blur_radius, l_subframe_size)

#set AOI
x_off = int(round(centroid[0])) - l_subframe_size
y_off = int(round(centroid[1])) - l_subframe_size
fgs_var.fgs_cam.disable_hotPixelCorrection()
fgs_var.fgs_cam.set_AOI_size(l_subframe_size*2, l_subframe_size*2)
fgs_var.fgs_cam.set_AOI_position(x_off, y_off)
image = fgs_var.get_frame() - bias
plt.imshow(image)

res = Utilities.centroid_iterative(image, convergence_limit=0.5, max_iterations=10, window_size=subframe_size,
                                       x_offset=x_off, y_offset=y_off)

#test speed and accuracy
centroid_gauss = np.zeros((100, 2))
centroid_mass = np.zeros((100, 2))
time_mass = 0
time_gauss = 0
time_repos = 0

for i in range(0, 100):
    image = fgs_var.get_frame() - bias

    t_i = time.time()
    res = Utilities.centroid_iterative(image, convergence_limit=0.5, max_iterations=10, window_size=subframe_size,
                                       x_offset=x_off, y_offset=y_off)
    t_i2 = time.time()
    res_gauss = Utilities.fit_gauss_elliptical([y_off, x_off], image)
    t_i3 = time.time()


    #Following AOI move causes problems!
    x_off_new = int(round(res_gauss[4])) - l_subframe_size
    y_off_new = int(round(res_gauss[3])) - l_subframe_size

    if (x_off_new - x_off > 5) or (y_off_new - y_off > 5):
        x_off = x_off_new
        y_off = y_off_new
        # fgs_var.fgs_cam.set_AOI_position(x_off, y_off)
        fgs_var.fgs_cam.set_AOI_position_fast(x_off, y_off)
        # needed after a fast AOI move
        time.sleep(0.003)
        image = fgs_var.get_frame() - bias

    t_e = time.time()

    centroid_gauss[i, 0] = res_gauss[4]
    centroid_gauss[i, 1] = res_gauss[3]
    centroid_mass[i, 0] = res[0]
    centroid_mass[i, 1] = res[1]

    time_gauss = time_gauss + (t_i3 - t_i2)
    time_mass = time_mass + (t_i2 - t_i)
    time_repos = time_repos + (t_e - t_i3)

print("Time needed gauss: %f ms" % ((time_gauss / 100.0) * 1000.0))
print("Time needed mass: %f ms" % ((time_mass / 100.0) * 1000.0))
print("Time needed repos: %f ms" % ((time_repos / 100.0) * 1000.0))
mass_rms = np.std(centroid_mass, 0)
print("rms mass (x, y): {} pix".format(mass_rms))
gauss_rms = np.std(centroid_gauss, 0)
print("rms gauss (x, y): {} pix".format(gauss_rms))
diff = np.mean(centroid_gauss, 0) - np.mean(centroid_mass, 0)
print("center diff (x, y): {} pix".format(diff))

plt.figure()
plt.scatter(centroid_gauss[:, 0], centroid_gauss[:, 1], c='coral')
plt.scatter(centroid_mass[:, 0], centroid_mass[:, 1],  c='lightblue')

fgs_var.close()


