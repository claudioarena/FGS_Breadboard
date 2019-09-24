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

def fit_gauss_elliptical(xy, data):
    """
    ---------------------
    Purpose
    Fitting a star with a 2D elliptical gaussian PSF.
    ---------------------
    Inputs
    * xy (list) = list with the form [x,y] where x and y are the integer positions in the complete image of the first pixel (the one with x=0 and y=0) of the small subimage that is used for fitting.
    * data (2D Numpy array) = small subimage, obtained from the full FITS image by slicing. It must contain a single object : the star to be fitted, placed approximately at the center.
    ---------------------
    Output (list) = list with 8 elements, in the form [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]. The list elements are respectively:
    - maxi is the value of the star maximum signal,
    - floor is the level of the sky background (fit result),
    - height is the PSF amplitude (fit result),
    - mean_x and mean_y are the star centroid x and y positions, on the full image (fit results),
    - fwhm_small is the smallest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - fwhm_large is the largest full width half maximum of the elliptical gaussian PSF (fit result) in pixels
    - angle is the angular direction of the largest fwhm, measured clockwise starting from the vertical direction (fit result) and expressed in degrees. The direction of the smallest fwhm is obtained by adding 90 deg to angle.
    ---------------------
    """

    #find starting values
    dat=data.flatten()
    maxi = data.max()
    floor = np.ma.median(dat)
    height = maxi - floor
    if height==0.0:             #if star is saturated it could be that median value is 32767 or 65535 --> height=0
        floor = np.mean(dat)
        height = maxi - floor

    mean_x = (np.shape(data)[0]-1)/2
    mean_y = (np.shape(data)[1]-1)/2

    fwhm = np.sqrt(np.sum((data>floor+height/2.).flatten()))
    fwhm_1 = fwhm
    fwhm_2 = fwhm
    sig_1 = fwhm_1 / (2.*np.sqrt(2.*np.log(2.)))
    sig_2 = fwhm_2 / (2.*np.sqrt(2.*np.log(2.)))

    angle = 0.

    p0 = floor, height, mean_x, mean_y, sig_1, sig_2, angle

    #---------------------------------------------------------------------------------
    #fitting gaussian
    def gauss(floor, height, mean_x, mean_y, sig_1, sig_2, angle):

        A = (np.cos(angle)/sig_1)**2. + (np.sin(angle)/sig_2)**2.
        B = (np.sin(angle)/sig_1)**2. + (np.cos(angle)/sig_2)**2.
        C = 2.0*np.sin(angle)*np.cos(angle)*(1./(sig_1**2.)-1./(sig_2**2.))

        #do not forget factor 0.5 in exp(-0.5*r**2./sig**2.)
        return lambda x,y: floor + height*np.exp(-0.5*(A*((x-mean_x)**2)+B*((y-mean_y)**2)+C*(x-mean_x)*(y-mean_y)))

    def err(p,data):
        return np.ravel(gauss(*p)(*np.indices(data.shape))-data)

    p = leastsq(err, p0, args=(data), maxfev=200)
    p = p[0]

    #---------------------------------------------------------------------------------
    #formatting results
    floor = p[0]
    height = p[1]
    mean_x = p[2] + xy[0]
    mean_y = p[3] + xy[1]

    #angle gives the direction of the p[4]=sig_1 axis, starting from x (vertical) axis, clockwise in direction of y (horizontal) axis
    if np.abs(p[4])>np.abs(p[5]):

        fwhm_large = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        angle = np.arctan(np.tan(p[6]))

    else:   #then sig_1 is the smallest : we want angle to point to sig_y, the largest

        fwhm_large = np.abs(p[5]) * (2.*np.sqrt(2.*np.log(2.)))
        fwhm_small = np.abs(p[4]) * (2.*np.sqrt(2.*np.log(2.)))
        angle = np.arctan(np.tan(p[6]+np.pi/2.))

    output = [maxi, floor, height, mean_x, mean_y, fwhm_small, fwhm_large, angle]
    return output


def centroid_iterative(image):
    # Now blur to find the peak
    blur_radius = 2
    pil_im = Image.fromarray(np.uint8(image), 'L')
    img_fgs_blur = pil_im.filter(ImageFilter.BoxBlur(blur_radius))
    blurred = np.array(img_fgs_blur)
    median = np.median(blurred)

    wsize = 15  # half width window
    first_guess = np.where(blurred == blurred.max())
    subframe = blurred[(first_guess[0][0] - wsize):(first_guess[0][0] + wsize),
               (first_guess[1][0] - wsize):(first_guess[1][0] + wsize)]
    # plt.imshow(subframe)

    centroid = ndimage.measurements.center_of_mass(subframe)
    cy = centroid[0] + (first_guess[0][0] - wsize)
    cx = centroid[1] + (first_guess[1][0] - wsize)

    diffX = 100
    diffY = 100
    n = 0
    while (diffX > 1 and diffY > 1) or (n < 10):
        old_cx = cx
        old_cy = cy

        try:
            subframe = blurred[int(round(cy) - wsize):int(round(cy) + wsize),
                       int(round(cx) - wsize):int(round(cx) + wsize)]
            centroid = ndimage.measurements.center_of_mass(subframe)
            cy = centroid[1] + int(round(cy) - wsize)
            cx = centroid[0] + int(round(cx) - wsize)

            diffX = abs(cx - old_cx)
            diffY = abs(cy - old_cy)
        except Exception:
            return [0, 0]

        n = n + 1

    return [cx, cy]


class fgs:
    controller = ""
    fgs_cam = ""

    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

    axis_sensitivity = [0, 0, 0] # pix per volt
    axis_angle = [0, 0, 0] # in radians, from positive x-coordinate axis
    axis_x_proj = [0, 0, 0] # projection, in volt/pix
    axis_y_proj = [0, 0, 0] # projection, in volt/pix

    def __init__(self, controller_port="", fgs_camera_id=2, useDevID=True):
        self.controller = Controller(controller_port)
        self.reset_controller_v()
        self.fgs_cam = uc480.uc480()
        try:
            self.fgs_cam.disconnect()
        except:
            pass
        self.fgs_cam.connect(fgs_camera_id, useDevID)

    def close(self):
        self.controller.close()
        self.fgs_cam.disconnect()

    def reset_controller_v(self):
        max_v = self.controller.get_sys_voltage_max()
        v = max_v / 2.0
        self.controller.set_xyz_voltage(v, v, v)

    def setup_camera(self):
        self.fgs_cam.set_gain(0)
        self.fgs_cam.set_gain_boost(False)
        minClock, maxClock, _ = self.fgs_cam.get_clock_limits()
        self.fgs_cam.set_blacklevel(uc480.IS_AUTO_BLACKLEVEL_ON)
        self.fgs_cam.set_clock(minClock)
        self.fgs_cam.set_exposure(60)
        self.fgs_cam.set_clock(24)
        self.fgs_cam.set_frameRate(15)
        # minFPS, maxFPS, _ = fgs_cam.get_Framerate_limits()
        # fgs_cam.set_clock(maxFPS)
        self.fgs_cam.set_exposure(60)
        print(self.fgs_cam.get_exposure())
        self.fgs_cam.acquire()
        self.fgs_cam.acquire()
        img_fgs = self.fgs_cam.acquire()

        #plt.imshow(img_fgs)
        #plt.show()
        #plt.draw()
        #plt.show()             #this plots correctly, but blocks execution.
        #plt.show(block=False)   #this creates an empty frozen window.
        plt.figure()
        plt.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
        plt.show()
        plt.pause(0.1)

        input("Press Enter to continue...")

    def get_frame(self):
        img_fgs = self.fgs_cam.acquire()
        return img_fgs


    def setv(self, vx, vy, vz):
        self.controller.set_xyz_voltage(vx, vy, vz)
        time.sleep(0.5)
        self.fgs_cam.acquire()
        self.fgs_cam.acquire()
        img_fgs = self.fgs_cam.acquire()
        cx, cy = centroid_iterative(img_fgs)
        return cx, cy

    def calibrate(self, n_average, v_start, v_end, v_n, pause_time):
        self._calibrate_axis(self.X_AXIS, n_average, v_start, v_end, v_n, pause_time)
        time.sleep(pause_time)
        self._calibrate_axis(self.Y_AXIS, n_average, v_start, v_end, v_n, pause_time)
        time.sleep(pause_time)
        self._calibrate_axis(self.Z_AXIS, n_average, v_start, v_end, v_n, pause_time)

    def _calibrate_axis(self, axis, n_average, v_start, v_end, v_step, pause_time):
        v_range = np.arange(v_start, v_end, v_step)
        cx = np.zeros(len(v_range), dtype=np.float)
        cy = np.zeros(len(v_range), dtype=np.float)
        self.fgs_cam.acquire()
        self.fgs_cam.acquire()

        if axis is self.X_AXIS:
            set_voltage = self.controller.set_x_voltage
        elif axis is self.Y_AXIS:
            set_voltage = self.controller.set_y_voltage
        elif axis is self.Z_AXIS:
            set_voltage = self.controller.set_z_voltage

        set_voltage(v_start)
        time.sleep(pause_time*4)
        self.fgs_cam.acquire()

        n = 0
        for V in v_range:
            print("voltage set to %d" % V)

            set_voltage(V)
            time.sleep(pause_time)
            self.fgs_cam.acquire()
            self.fgs_cam.acquire()

            for i in range(0, n_average):
                img_fgs = self.fgs_cam.acquire()
                cx_temp, cy_temp = centroid_iterative(img_fgs)
                cx[n] = cx[n] + cx_temp
                cy[n] = cy[n] + cy_temp

            cx[n] = cx[n] / n_average
            cy[n] = cy[n] / n_average
            n = n + 1

        self.reset_controller_v()

        fit = np.polyfit(cx, cy, 1)
        d = np.sqrt(np.square(cx)+np.square(cy))
        fit_v = np.polyfit(v_range, d, 1)

        self.axis_angle[axis] = math.atan(fit[0])
        self.axis_sensitivity[axis] = fit_v[0]
        self.axis_x_proj[axis] = (math.cos(self.axis_angle[axis])) / self.axis_sensitivity[axis]
        self.axis_y_proj[axis] = (math.sin(self.axis_angle[axis])) / self.axis_sensitivity[axis]

        p = np.poly1d(fit)
        r2 = metrics.r2_score(cy, p(cx))
        p_v = np.poly1d(fit_v)
        r2_v = metrics.r2_score(d, p_v(v_range))

        plt.figure()
        plt.scatter(cx, cy)
        plt.figure()
        plt.scatter(v_range, d)
        input("Press Enter to continue...")

        #return [cx, cy]
