import numpy as np
from FGS_Breadboard import fgs
import matplotlib.pyplot as plt

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482", exp_time=13)
fgs_var.set_voltages(50, 50, 50)
#image = fgs_var.get_frame(bias_correct=True)
#plt.imshow(image)

# PID
fgs_var.axis_x_proj = [0.114, -0.312949, -0.01793]  # projection, in volt/pix
fgs_var.axis_y_proj = [0.0723, 0.19102, -0.1958]  # projection, in volt/pix

pos, errors, t = fgs_var.pid_start(750, 601, repetitions=100)
plt.figure()
plt.scatter(t, errors[:, 0],  c='coral')
plt.scatter(t, errors[:, 1],  c='lightblue')
plt.figure()
plt.scatter(t, pos[:, 0],  c='coral')
plt.scatter(t, pos[:, 1],  c='lightblue')

fgs_var.close()

"""
#test speed and accuracy
centroid_gauss = np.zeros((100, 2))
centroid_mass = np.zeros((100, 2))
time_mass = 0
time_gauss = 0
time_repos = 0

print("Time needed gauss: %f ms" % ((time_gauss / 100.0) * 1000.0))
print("Time needed mass: %f ms" % ((time_mass / 100.0) * 1000.0))
print("Time needed repos: %f ms" % ((time_repos / 100.0) * 1000.0))
mass_rms = np.std(centroid_mass, 0)
print("rms mass (x, y): {} pix".format(mass_rms))
gauss_rms = np.std(centroid_gauss, 0)
print("rms gauss (x, y): {} pix".format(gauss_rms))
diff = np.mean(centroid_gauss, 0) - np.mean(centroid_mass, 0)
print("center diff (x, y): {} pix".format(diff))

"""

