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


class Utilities:

    @staticmethod
    def center_of_mass(image, threshold=-2000):
        im = image
        im[im < threshold] = 0
        shape = np.shape(im)
        # Summed such that each value in this array is the sum of all y pixels for a given x position
        # i.e. same size as x dimension of image (x and y according to plt.imshow visual)
        xSum = np.sum(im, axis=0)
        # Summed such that each value in this array is the sum of all x pixels for a given y position
        ySum = np.sum(im, axis=1)

        tot = np.sum(ySum)
        xCenter = np.sum(np.multiply(xSum, np.arange(shape[1]))) / tot
        yCenter = np.sum(np.multiply(ySum, np.arange(shape[0]))) / tot
        return xCenter, yCenter

    @staticmethod
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
        if height is 0.0:             #if star is saturated it could be that median value is 32767 or 65535 --> height=0
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

    @staticmethod
    # This finds a rough estimate of the centroid (within a few pixels) from a whole image
    def centroid_approximate(image, blur_radius=5, subframe_size=10):
        # Now blur to find the peak
        subframe_size = int(round(subframe_size))
        offset_image = (image - image.min())
        if offset_image.max() > 255:
            offset_image = offset_image / (offset_image.max() / 255.0)

        pil_im = Image.fromarray(np.uint8(offset_image), 'L')
        img_fgs_blur = pil_im.filter(ImageFilter.BoxBlur(blur_radius))
        blurred = np.array(img_fgs_blur)
        median = np.median(blurred)

        first_guess = np.where(blurred == blurred.max())
        subframe = image[(first_guess[0][0] - subframe_size):(first_guess[0][0] + subframe_size),
                   (first_guess[1][0] - subframe_size):(first_guess[1][0] + subframe_size)]
        # plt.imshow(subframe)

        cent = Utilities.center_of_mass(subframe)
        cy = cent[1] + first_guess[0][0] - subframe_size
        cx = cent[0] + first_guess[1][0] - subframe_size
        return [cx, cy]

    @staticmethod
    # Feed a small subframe to this, and it will iteratively try to find the centroid.
    # Set the x_offset and y_offset if you would like the returned centroid to be referred to the original full image.
    def centroid_iterative(image, x_offset=0, y_offset=0, convergence_limit=0.5, max_iterations=10, window_size=10):
        # centroid should refer to respect to image coordinate (which would normally be a subframe)
        # We can then offset to full image (with x_offset and y_offset) if needed.
        window_size = int(round(window_size))
        centroid = Utilities.center_of_mass(image)
        old_centroid = centroid
        diff = np.array([5.0*window_size, 5.0*window_size])
        n = 0

        while (diff.max() > convergence_limit) and (n < max_iterations):
            old_centroid = centroid

            try:
                x_off = int(round(centroid[0])) - window_size
                y_off = int(round(centroid[1])) - window_size
                if x_off < 0:
                    x_off = 0
                if y_off < 0:
                    y_off = 0

                x_end = x_off + 2 * window_size + 1
                y_end = y_off + 2 * window_size + 1
                if x_end > image.shape[0]:
                    x_end = image.shape[0]
                    x_off = image.shape[0] - (2 * window_size + 1)
                if y_end > image.shape[1]:
                    y_end = image.shape[1]
                    y_off = image.shape[1] - (2 * window_size + 1)

                subframe = image[y_off:y_end, x_off:x_end]
                centroid = Utilities.center_of_mass(subframe)
                centroid = np.add(centroid, [x_off, y_off])
                diff = np.absolute(np.subtract(np.round(centroid), np.round(old_centroid)))

            except Exception:
                return [0, 0]

            n = n + 1

        cx = centroid[0] + x_offset
        cy = centroid[1] + y_offset
        return cx, cy, diff[0], diff[1]


    @staticmethod
    ## image is searched for a max, then centroid determined using center of mass on a spot_frame_size windows
    #spot_frame_size windows shouldn't be too large, or else result might be too skewed.
    ## Then a spot_frame_size windows is used to iteractively find the centroid.
    ## For iteractive search, a subframe_size subframe is used
    def initial_centroid(image, blur_radius=5, subframe_size=10, spot_frame_size=5):
        subframe_size = int(round(subframe_size))
        spot_frame_size = int(round(spot_frame_size))

        centroid = Utilities.centroid_approximate(image, blur_radius, spot_frame_size * 2.0)
        x_off = int(round(centroid[0])) - subframe_size
        y_off = int(round(centroid[1])) - subframe_size
        subimage = image[y_off:y_off + 2*subframe_size+1, x_off:x_off + 2*subframe_size+1]
        res = Utilities.centroid_iterative(subimage, convergence_limit=0.5, max_iterations=10, window_size=spot_frame_size,
                                           x_offset=x_off, y_offset=y_off)
        return res[0], res[1]


class fgs:
    controller = ""
    fgs_cam = ""
    bias_level = 0

    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2
    fwhm_x = 0
    fwhm_y = 0

    x_offset = 0
    y_offset = 0

    fgs_cam_AOI_pos_inc = [0, 0]
    fgs_cam_AOI_size_inc = [0, 0]

    axis_sensitivity = [0, 0, 0] # pix per volt
    axis_angle = [0, 0, 0] # in radians, from positive x-coordinate axis
    axis_x_proj = [0, 0, 0] # projection, in volt/pix
    axis_y_proj = [0, 0, 0] # projection, in volt/pix

    axis_sensitivity = [9.830764734822177, -4.152147528750609, -6.103315367191801] # pix per volt
    axis_angle = [0.5013038217873975, -0.4916111054237689, 1.5564821039078214] # in radians, from positive x-coordinate axis
    axis_x_proj = [0.08920534203964842, -0.21231747662981001, -0.0023452391376540906] # projection, in volt/pix
    axis_y_proj = [0.04888422772871696, 0.11368738821278822, -0.16382859038150982] # projection, in volt/pix

    pid_P = 0.4
    pid_I = 0
    pid_D = 0.0

    def __init__(self, controller_port="", fgs_camera_SerNo="", exp_time=10, gain=0, gain_boost=False):
        try:
            self.controller = Controller(controller_port)
            self.reset_controller_v()
            self.fgs_cam = uc480.uc480()

            try: # try disconnecting before connecting
                self.fgs_cam.disconnect()
            except Exception:
                pass

            if(fgs_camera_SerNo is not ""):
                self.fgs_cam.connect_with_SerNo(SerNo=fgs_camera_SerNo)
            else:
                self.fgs_cam.connect_with_SerNo()

            self.setup_camera(exp_time=exp_time, gain=gain, gain_boost=gain_boost)

        except Exception:
            self.controller.close()
            self.fgs_cam.disconnect()
            pass

    def close(self):
        try:
            self.controller.close()
        except Exception:
            pass

        try:
            self.fgs_cam.disconnect()
        except Exception:
            pass

    def reset_controller_v(self):
        max_v = self.controller.get_sys_voltage_max()
        v = max_v / 2.0
        self.controller.set_xyz_voltage(v, v, v)

    # Exposure time in ms
    def setup_camera(self, exp_time, gain=0, gain_boost=False):
        self.fgs_cam.set_gain(gain)
        self.fgs_cam.set_gain_boost(gain_boost)
        minClock, maxClock, _ = self.fgs_cam.get_clock_limits()
        self.fgs_cam.set_blacklevel(uc480.IS_AUTO_BLACKLEVEL_ON)
        self.fgs_cam.set_clock(minClock)
        self.fgs_cam.set_exposure(exp_time)
        self.fgs_cam.set_clock(24)
        self.fgs_cam.set_framerate(15)
        self.fgs_cam.disable_hotPixelCorrection()
        self.fgs_cam.reset_AOI()

        self.fgs_cam_AOI_pos_inc = self.fgs_cam.get_AOI_pos_inc()
        self.fgs_cam_AOI_size_inc = self.fgs_cam.get_AOI_size_inc()
        # minFPS, maxFPS, _ = fgs_cam.get_Framerate_limits()
        # fgs_cam.set_clock(maxFPS)
        self.fgs_cam.set_exposure(exp_time)
        #print(self.fgs_cam.get_exposure())
        #self.fgs_cam.acquire()
        img_fgs = self.fgs_cam.acquire()

        #plt.imshow(img_fgs)
        #plt.show()
        #plt.draw()
        #plt.show()             #this plots correctly, but blocks execution.
        #plt.show(block=False)   #this creates an empty frozen window.
        #plt.figure()
        #plt.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
        #plt.show()
        #plt.pause(0.1)

        print("Image max value: %f" % img_fgs.max())
        #input("Press Enter to continue...")

        centroid = Utilities.centroid_approximate(img_fgs, blur_radius=5, subframe_size=40)
        x_off = int(round(centroid[0])) - 40
        y_off = int(round(centroid[1])) - 40
        subimage = img_fgs[y_off:y_off + 2*40+1, x_off:x_off + 2*40+1]
        res = Utilities.fit_gauss_elliptical([x_off, y_off], subimage)
        self.fwhm_x = res[6]
        self.fwhm_y = res[6]
        self.get_bias_level()

        # TODO: from large / small FWHM and angle to x and y projections, if needed
        # fwhm_x = res[5] * math.sin(math.radians(res[7]))
        # fwhm_y = res[6] * res[7]

    # Average all the pixels apart from the 80x80 window around the suspected spot
    def get_bias_level(self):
        im = self.get_frame()
        cx, cy = Utilities.centroid_approximate(im, blur_radius=5, subframe_size=10)
        windowSize = 40

        average = (im.sum() - im[int(round(cy - windowSize)):int(round(cy + windowSize)),
                              int(round(cx - windowSize)):int(round(cx + windowSize))].sum()) / im.size
        self.bias_level = average
        return self.bias_level

    # Average all the pixel on the border. Useful when already using a small window around the spot
    def get_bias_level_method2(self):
        im = self.get_frame()
        total = np.sum(im[:, 0]) + np.sum(im[0, :]) + np.sum(im[:, -1]) + np.sum(im[-1, :])
        n_pixels = 2*im.shape[0] + 2*im.shape[1]
        self.bias_level = total / n_pixels
        return self.bias_level

    def get_frame(self, bias_correct=False):
        img_fgs = self.fgs_cam.acquire()
        if bias_correct:
            img_fgs = img_fgs - self.bias_level
        return img_fgs

    # Calculate the three piezo voltage offset needed, for a given dx dy position wanted
    # Using small angle approximation, and using the calibration values.
    # Calibration values (divided by two, since we're using all three piezo, in a push pull arrangement)
    # are in V/p. So multiplying it by the needed offset produces the needed Voltage offsets needed.
    def calculate_piezo_offset(self, delta_x, delta_y):
        deltaVX = delta_x * (self.axis_x_proj[0] / 2.0) + delta_y * (self.axis_y_proj[0] / 2.0)
        deltaVY = delta_x * (self.axis_x_proj[1] / 2.0) + delta_y * (self.axis_y_proj[1] / 2.0)
        deltaVZ = delta_x * (self.axis_x_proj[2] / 2.0) + delta_y * (self.axis_y_proj[2] / 2.0)
        return [deltaVX, deltaVY, deltaVZ]

    def set_voltages(self, vx, vy, vz):
        self.controller.set_xyz_voltage(vx, vy, vz)

    def reset_AOI(self):
        self.fgs_cam.reset_AOI()

    def update_AOI(self, fast=True, min_diff_threshold=2, image=0, centroid=0, spot_frame_size=1.5):
        if image is 0:
            image = self.get_frame(bias_correct=True)

        if centroid is 0:
            centroid = self.find_centroid(image, convergence_limit=0.5, max_iterations=10, window_size=spot_frame_size)

        diff_x = int(round(centroid[0])) - int(round(image.shape[0]/2.0))
        diff_y = int(round(centroid[1])) - int(round(image.shape[1]/2.0))

        ## The following makes sure that, if not a multiple of the increments allowed, it rounds to the smallest allowed value
        if diff_x % self.fgs_cam_AOI_pos_inc[0] is not 0:
            diff_x = int(np.trunc(diff_x / self.fgs_cam_AOI_pos_inc[0]) * self.fgs_cam_AOI_pos_inc[0])
        if diff_y % self.fgs_cam_AOI_pos_inc[1] is not 0:
            diff_y = int(np.trunc(diff_y / self.fgs_cam_AOI_pos_inc[1]) * self.fgs_cam_AOI_pos_inc[1])

        ## Threshold min AOI move
        if self.fgs_cam_AOI_pos_inc[0] > min_diff_threshold:
            min_x_diff = self.fgs_cam_AOI_pos_inc[0]
        else:
            min_x_diff = min_diff_threshold
        if self.fgs_cam_AOI_pos_inc[1] > min_diff_threshold:
            min_y_diff = self.fgs_cam_AOI_pos_inc[1]
        else:
            min_y_diff = min_diff_threshold

        if abs(diff_x) >= min_x_diff or abs(diff_y) >= min_y_diff:
            self.x_offset = self.x_offset + diff_x
            self.y_offset = self.y_offset + diff_y
            if fast:
                self.fgs_cam.set_AOI_position_fast(self.x_offset, self.y_offset)
            else:
                self.fgs_cam.set_AOI_position(self.x_offset, self.y_offset)

        return self.x_offset, self.y_offset, image

    ## subframe_size and spot_frame_size are in fwhm units
    def setup_AOI(self, half_width, half_height, spot_frame_size=1.5):
        half_width = int(round(half_width))
        half_height = int(round(half_height))
        self.fgs_cam.reset_AOI()
        image = self.get_frame(bias_correct=True)
        blur_radius = 5
        subframe_size = np.min([half_width, half_height])
        centroid = self.find_centroid_blind(blur_radius,
                subframe_size=subframe_size, spot_frame_size=spot_frame_size)
        self.x_offset, self.y_offset = self.set_AOI(centroid[0], centroid[1], half_width, half_height)
        return self.x_offset, self.y_offset

    def set_AOI(self, x_center, y_center, half_width, half_height):
        new_offset_x = int(round(x_center)) - half_width
        new_offset_x = new_offset_x - (new_offset_x % self.fgs_cam_AOI_pos_inc[0])
        new_offset_y = int(round(y_center)) - half_height
        new_offset_y = new_offset_y - (new_offset_y % self.fgs_cam_AOI_pos_inc[1])

        new_AOI_width = half_width * 2 + 1
        new_AOI_height = half_height * 2 + 1
        # The below makes sure that, if not a multiple of the size increment, the width and height are the next possible step up
        if new_AOI_width % self.fgs_cam_AOI_size_inc[0] is not 0:
            new_AOI_width = int((np.trunc(new_AOI_width / self.fgs_cam_AOI_size_inc[0]) + 1) * self.fgs_cam_AOI_size_inc[0])
        if new_AOI_height % self.fgs_cam_AOI_size_inc[1] is not 0:
            new_AOI_height = int((np.trunc(new_AOI_height / self.fgs_cam_AOI_size_inc[1]) + 1) * self.fgs_cam_AOI_size_inc[1])

        self.x_offset = new_offset_x
        self.y_offset = new_offset_y
        self.fgs_cam.set_AOI_size(new_AOI_width, new_AOI_height)
        self.fgs_cam.set_AOI_position(self.x_offset, self.y_offset)
        return self.x_offset, self.y_offset

    ##Overload for constant target
    def pid_start_const_target(self, target_x, target_y, repetitions, timeout, output_active=True, subFrame_size=20, spot_frame_size=1.5):

        def const_x(time):
            return target_x

        def const_y(time):
            return target_y

        return self.pid_start(const_x, const_y, repetitions, timeout, output_active, subFrame_size, spot_frame_size)

    #Subframe size is in pix, spot_frame_size is in multiples of the FWHM
    def pid_start(self, func_target_x, func_target_y, repetitions, timeout, output_active=True, subFrame_size=20, spot_frame_size=1.5):
        set_v = [50.0, 50.0, 50.0]
        #set_v = self.controller.get_xyz_voltage()

        if output_active:
            self.set_voltages(*set_v)
        time.sleep(5)

        x_off, y_off = self.setup_AOI(subFrame_size, subFrame_size, spot_frame_size)

        # If centroid is not in the center of the subframe, adjust subframe
        # but only do it after the distance is greater that these thresholds:
        min_mov_x = self.fwhm_x / 3.0
        min_mov_x = 10 if min_mov_x < 10 else min_mov_x
        min_mov_y = self.fwhm_y / 3.0
        min_mov_y = 10 if min_mov_y < 10 else min_mov_y

        pid_x = PID(self.pid_P, self.pid_I, self.pid_D, setpoint=func_target_x(0))
        pid_y = PID(self.pid_P, self.pid_I, self.pid_D, setpoint=func_target_y(0))
        pid_x.output_limits = (-self.fwhm_x * 1.5, self.fwhm_x * 1.5) # in pixels
        pid_y.output_limits = (-self.fwhm_x * 1.5, self.fwhm_x * 1.5)

        errs = np.zeros((repetitions, 2))
        pos = np.zeros((repetitions, 2))
        t = np.zeros(repetitions)
        total_light = np.zeros(repetitions)
        slit_light = np.zeros(repetitions)

        # self.fgs_cam.set_AOI_size(40, 40)
        # self.fgs_cam.set_AOI_position(672, 566)
        # self.fgs_cam.set_clock(self.fgs_cam.get_clock_limits()[1])
        # self.fgs_cam.set_exposure(5)
        # self.fgs_cam.set_framerate(100)
        self.get_bias_level_method2()
        im = self.get_frame(bias_correct=True)
        tot_weight = im.sum()
        #time.sleep(1)
        plt.imshow(im)
        plt.show()

        cx_ini, cy_ini = self.find_centroid(im, x_offset=0, y_offset=0, convergence_limit=0.5,
                                    max_iterations=10, window_size=spot_frame_size)
        cx_ini = int(round(cx_ini))
        cy_ini = int(round(cy_ini))
        slit_halfsize = int(round(self.fwhm_x * spot_frame_size / 2.0))

        print("Slit half size: {}".format(slit_halfsize))
        #Withot the image acquisition, and output off, the cycle below can be performed
        # ~ 3000 times per second on my laptop. So speed is limited by frame rate
        # Almost all of the delay (~ 0.035 sec in 100 repetitions) comes from the find_centroid method.
        # The update_AOI, without a centroid param, also has a second find_centroid.
        t_0 = time.time()

        for i in range(0, repetitions):
            t[i] = time.time()
            img_fgs = self.get_frame(bias_correct=True)

            total_light[i] = img_fgs.sum()
            slit_im = img_fgs[cx_ini-slit_halfsize:cx_ini+slit_halfsize,
                      cy_ini-slit_halfsize:cy_ini+slit_halfsize]
            slit_light[i] = np.sum(slit_im)

            if img_fgs.sum() < (tot_weight / 2.0):
                print("Low weight")
                x_off, y_off = self.setup_AOI(subFrame_size, subFrame_size, spot_frame_size)
                img_fgs = self.get_frame(bias_correct=True)

            cx, cy = self.find_centroid(img_fgs, x_offset=x_off, y_offset=y_off, convergence_limit=0.5,
                                        max_iterations=10, window_size=spot_frame_size)

            errs[i, :] = [cx - func_target_x(t[i]-t_0), cy - func_target_y(t[i]-t_0)]
            pos[i, :] = [cx, cy]

            if output_active:
                pid_x.setpoint = func_target_x(t[i]-t_0)
                pid_y.setpoint = func_target_y(t[i]-t_0)
                x_control = pid_x(cx)
                y_control = pid_y(cy)
                delta_voltages = self.calculate_piezo_offset(x_control, y_control)
                set_v[0] = set_v[0] + delta_voltages[0]
                set_v[1] = set_v[1] + delta_voltages[1]
                set_v[2] = set_v[2] + delta_voltages[2]
                self.set_voltages(*set_v)

            centroid = [cx - x_off, cy - y_off]
            x_off, y_off, _ = self.update_AOI(image=img_fgs, min_diff_threshold=np.max([min_mov_x, min_mov_y]),
                                              centroid=centroid, spot_frame_size=spot_frame_size)

            if (t[i]-t_0) > timeout:
                break

        norm_t = t-t_0
        pos = pos[:i-1, :]
        errs = errs[:i-1, :]
        norm_t = norm_t[:i-1]
        return pos, errs, norm_t, total_light, slit_light

    def test_speed(self):
        self.fgs_cam.set_AOI_size(20, 20)
        self.fgs_cam.set_AOI_position(500, 500)
        self.fgs_cam.set_clock(self.fgs_cam.get_clock_limits()[1])
        self.fgs_cam.set_exposure(1.0)
        self.fgs_cam.set_framerate(1000/3.0)

        print("FPS: %f" % self.fgs_cam.get_framerate())
        print("clock: %f" % self.fgs_cam.get_clock())
        print("exp: %f" % self.fgs_cam.get_exposure())
        #self.fgs_cam.set_framerate(200)

        # test FPS with AOI:
        t_i = time.time()

        for n in range(0, 100):
            img_fgs = self.fgs_cam.acquire()
            #self.calculate_piezo_offset(0, 0)
            #self.set_voltages(50, 50, 50)

        t_e = time.time()
        print("Time needed: %f" % (t_e - t_i))
        print("FPS: %f" % (100 / (t_e - t_i)))

        self.fgs_cam.reset_AOI()
        return img_fgs

    def find_centroid_blind(self, blur_radius=5, subframe_size=10, spot_frame_size=1.5):
        self.fgs_cam.reset_AOI()
        img_fgs = self.fgs_cam.acquire() - self.bias_level
        cx, cy = Utilities.initial_centroid(img_fgs, blur_radius, subframe_size, spot_frame_size * self.fwhm_x)
        return [cx, cy]

    def find_centroid(self, image, x_offset=0, y_offset=0, convergence_limit=0.5, max_iterations=10, window_size=1.5):
        window_size = window_size * self.fwhm_x
        cx, cy, _, _ = Utilities.centroid_iterative(image, x_offset=x_offset, y_offset=y_offset,
                                                    convergence_limit=convergence_limit, max_iterations=max_iterations,
                                                    window_size=window_size)
        return [cx, cy]

    def find_centroid_with_acquire(self, x_offset=0, y_offset=0, window_size=10):
        img_fgs = self.fgs_cam.acquire() - self.bias_level
        return self.find_centroid(img_fgs, x_offset, y_offset, window_size)

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
        #self.fgs_cam.acquire()
        set_voltage = ""

        if axis is self.X_AXIS:
            set_voltage = self.controller.set_x_voltage
        elif axis is self.Y_AXIS:
            set_voltage = self.controller.set_y_voltage
        elif axis is self.Z_AXIS:
            set_voltage = self.controller.set_z_voltage

        set_voltage(v_start)
        time.sleep(pause_time*4)
        #self.fgs_cam.acquire()

        n = 0
        for V in v_range:
            print("voltage set to %d" % V)

            set_voltage(V)
            time.sleep(pause_time)
            #self.fgs_cam.acquire()
            #self.fgs_cam.acquire()

            for i in range(0, n_average):
                #img_fgs = self.fgs_cam.acquire() - self.bias_level
                cx_temp, cy_temp = self.find_centroid_blind(blur_radius=5,
                subframe_size=20, spot_frame_size=1.5*self.fwhm_x)

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

        return cx, cy, v_range, d
