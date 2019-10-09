from FGS_Breadboard import fgs
from FGS_Breadboard import Utilities
import matplotlib.pyplot as plt

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482")
fgs_var.setup_camera(20)
bias = fgs_var.get_bias_level()
image = fgs_var.get_frame() - bias

blur_radius = 5
subframe_size = 8
l_subframe_size = subframe_size * 4
centroid = Utilities.centroid_approximate(image, blur_radius, l_subframe_size)
x_off = int(round(centroid[0])) - l_subframe_size
y_off = int(round(centroid[1])) - l_subframe_size
subimage = image[y_off:y_off + 2 * l_subframe_size, x_off:x_off + 2 * l_subframe_size]
plt.imshow(subimage)

res = Utilities.centroid_iterative(subimage, convergence_limit=0.5, max_iterations=10, window_size=subframe_size,
                                       x_offset=x_off, y_offset=y_off)

x_off2 = int(round(centroid[0])) - subframe_size
y_off2 = int(round(centroid[1])) - subframe_size
subimage2 = image[y_off2:y_off2 + 2 * subframe_size, x_off2:x_off2 + 2 * subframe_size]
plt.figure()
plt.imshow(subimage2)
res_gauss = Utilities.fit_gauss_elliptical([y_off2, x_off2], subimage2)
centroid[0] = res_gauss[4]
centroid[1] = res_gauss[3]

fgs_var.close()
