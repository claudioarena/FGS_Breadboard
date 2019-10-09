import uc480
import pylab as pl
from astropy.io import fits
import FGS_Breadboard
import time

fgs_cam = uc480.uc480()
fgs_cam.connect_with_SerNo(SerNo="4102821482")

fgs_cam.set_gain(0)
fgs_cam.set_gain_boost(False)

minClock, maxClock, _ = fgs_cam.get_clock_limits()
fgs_cam.set_clock(minClock)
fgs_cam.set_exposure(20)
fgs_cam.set_blacklevel(uc480.IS_AUTO_BLACKLEVEL_ON)
#fgs_cam.set_clock(7)
#fgs_cam.set_frameRate(1/(4/1000))
fgs_cam.set_exposure(20)
#minFPS, maxFPS, _ = fgs_cam.get_Framerate_limits()
#fgs_cam.set_clock(maxFPS)

# take a single image
img_fgs = fgs_cam.acquire()
pl.figure()
pl.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
pl.imshow(img_fgs)
print("Image max value: %f" % img_fgs.max())
pl.show()

# take a single image, subframe
x, y = FGS_Breadboard.centroid_iterative(img_fgs)
fgs_cam.set_AOI_size(20, 20)
fgs_cam.set_AOI_position(int(round(x-(20/2))), int(round(y-(20/2))))
img_fgs = fgs_cam.acquire()
pl.figure()
pl.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
pl.imshow(img_fgs)
print("Image max value: %f" % img_fgs.max())
pl.show()

#test FPS with AOI:
t_i = time.time()
for i in range(0, 100):
    img_fgs = fgs_cam.acquire()

t_e = time.time()
print("Time needed: %f" % (t_e - t_i))
print("FPS: %f" % (100/(t_e - t_i)))

fgs_cam.disconnect()

#fits.writeto(filename="data/FGS_image.fit", data=img_fgs, overwrite=True)

