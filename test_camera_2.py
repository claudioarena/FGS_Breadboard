import uc480
import pylab as pl
from astropy.io import fits

fgs_cam = uc480.uc480()
spectra_cam = uc480.uc480()
fgs_cam.connect(0)
spectra_cam.connect(1)

spectra_cam.set_gain(100)
spectra_cam.set_gain_boost(True)
minClock, maxClock, _ = spectra_cam.get_clock_limits()
spectra_cam.set_clock(minClock)
minFPS, maxFPS, _ = spectra_cam.get_Framerate_limits()
spectra_cam.set_frameRate(minFPS)
minExp, max_Exp, _ = spectra_cam.get_exposure_limits()
spectra_cam.set_exposure(max_Exp)

fgs_cam.set_gain(0)
fgs_cam.set_gain_boost(False)

minClock, maxClock, _ = fgs_cam.get_clock_limits()
fgs_cam.set_clock(minClock)
fgs_cam.set_exposure(68)
fgs_cam.set_clock(7)
fgs_cam.set_frameRate(1/(68/1000))

#minFPS, maxFPS, _ = fgs_cam.get_Framerate_limits()
#fgs_cam.set_clock(maxFPS)

# take a single image
img_spectra = spectra_cam.acquire(30)
img_fgs = fgs_cam.acquire()

fgs_cam.disconnect()
spectra_cam.disconnect()

fits.writeto(filename="FGS_image.fit", data=img_fgs, overwrite=True)
fits.writeto(filename="Spectra_image.fit", data=img_spectra, overwrite=True)

pl.imshow(img_spectra, cmap='gray', vmin=img_spectra.min(), vmax=img_spectra.max())
pl.figure()
pl.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
pl.show()

