from FGS_Breadboard import fgs
import time
import matplotlib.pyplot as plt
import pickle

fgs_var = fgs("COM5", fgs_camera_SerNo="4102821482")

with open(filename, ‘wb’) as f:
    pickle.dump(your_content, f)

import pickle
with open(filename, ‘wb’) as f:
    pickle.dump(your_content, f)

try:
    fgs_var.setup_camera(20)

    # n_average = 30
    # v_start = 42
    # v_end = 45
    # v_step = 0.2
    # pauseTime = 0.5
    #fgs_var.calibrate(n_average, v_start, v_end, v_step, pauseTime)

    fgs_var.axis_x_proj = [0.114, -0.312949, -0.01793]  # projection, in volt/pix
    fgs_var.axis_y_proj = [0.0723, 0.19102, -0.1958]  # projection, in volt/pix

    # fgs_var.set_voltages(50, 50, 50)
    #     # time.sleep(2)
    #     # cx, cy = fgs_var.find_centroid()
    #     # print("cx: %f cy:%f" % (cx, cy))
    #     #
    #     # dxv, dyv, dzv = fgs_var.calculate_piezo_offset(0, 20)
    #     # time.sleep(0.5)
    #     # fgs_var.set_voltages(50 + dxv, 50 + dyv, 50 + dzv)
    #     # time.sleep(0.5)
    #     # cx, cy = fgs_var.find_centroid()
    #     # print("cx: %f cy:%f" % (cx, cy))

    errors, t = fgs_var.pid_start()
    plt.figure()
    plt.scatter(range(0, 100), errors[:, 0])
    plt.figure()
    plt.scatter(range(0, 100), errors[:, 1])
    plt.figure()

    plt.plot(t)
    #plt.plot(errors)

finally:
    fgs_var.close()

#subframe = img_fgs[int(round(cx)-10):int(round(cx)+10), int(round(cy)-10):int(round(cy)+10)]
#pl.imshow(subframe)
#pl.scatter(x=(cy-int(round(cy)))+10, y=(cx-int(round(cx)))+10, c='r', s=40)

#pl.figure()
#pl.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
#pl.show()


