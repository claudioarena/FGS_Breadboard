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

        cent = ndimage.measurements.center_of_mass(subframe)
        cy = cent[0] + first_guess[0][0] - subframe_size
        cx = cent[1] + first_guess[1][0] - subframe_size
        return [cx, cy]

    @staticmethod
    # Feed a small subframe to this, and it will iteratively try to find the centroid.
    # Set the x_offset and y_offset if you would like the returned centroid to be referred to the original full image.
    def centroid_iterative(image, x_offset=0, y_offset=0, convergence_limit=0.5, max_iterations=10, window_size=10):
        # centroid should refer to respect to image coordinate (which would normally be a subframe)
        # We can then offset to full image (with x_offset and y_offset) if needed.
        window_size = int(round(window_size))
        centroid = np.array(ndimage.measurements.center_of_mass(image))
        old_centroid = centroid
        diff = np.array([5.0*window_size, 5.0*window_size])
        n = 0

        while (diff.max() > convergence_limit) and (n < max_iterations):
            old_centroid = centroid

            try:
                y_off = int(round(centroid[0])) - window_size
                x_off = int(round(centroid[1])) - window_size
                subframe = image[y_off:y_off + 2*window_size + 1, x_off:x_off + 2*window_size+1]
                centroid = np.array(ndimage.measurements.center_of_mass(subframe))
                centroid = np.add(centroid, [y_off, x_off])
                diff = np.absolute(np.subtract(np.round(centroid), np.round(old_centroid)))

            except Exception:
                return [0, 0]

            n = n + 1

        cy = centroid[0] + y_offset
        cx = centroid[1] + x_offset
        return cx, cy, diff[0], diff[1]


    @staticmethod
    def initial_centroid(image, blur_radius=5, subframe_size=10):
        subframe_size = int(round(subframe_size))
        l_subframe_size = subframe_size * 4
        centroid = Utilities.centroid_approximate(image, blur_radius, l_subframe_size)
        x_off = int(round(centroid[0])) - l_subframe_size
        y_off = int(round(centroid[1])) - l_subframe_size
        subimage = image[y_off:y_off + 2*l_subframe_size+1, x_off:x_off + 2*l_subframe_size+1]
        res = Utilities.centroid_iterative(subimage, convergence_limit=0.5, max_iterations=10, window_size=subframe_size,
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

    axis_sensitivity = [0, 0, 0] # pix per volt
    axis_angle = [0, 0, 0] # in radians, from positive x-coordinate axis
    axis_x_proj = [0, 0, 0] # projection, in volt/pix
    axis_y_proj = [0, 0, 0] # projection, in volt/pix

    def __init__(self, controller_port="", fgs_camera_SerNo="", exp_time=10):
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

            self.setup_camera(exp_time=exp_time)

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
    def setup_camera(self, exp_time):
        self.fgs_cam.set_gain(0)
        self.fgs_cam.set_gain_boost(False)
        minClock, maxClock, _ = self.fgs_cam.get_clock_limits()
        self.fgs_cam.set_blacklevel(uc480.IS_AUTO_BLACKLEVEL_ON)
        self.fgs_cam.set_clock(minClock)
        self.fgs_cam.set_exposure(exp_time)
        self.fgs_cam.set_clock(24)
        self.fgs_cam.set_framerate(15)
        self.fgs_cam.disable_hotPixelCorrection()
        # minFPS, maxFPS, _ = fgs_cam.get_Framerate_limits()
        # fgs_cam.set_clock(maxFPS)
        self.fgs_cam.set_exposure(exp_time)
        print(self.fgs_cam.get_exposure())
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

        # TODO: from large / small FWHM and angle to x and y projections, if needed
        # fwhm_x = res[5] * math.sin(math.radians(res[7]))
        # fwhm_y = res[6] * res[7]

    def get_bias_level(self):
        im = self.get_frame()
        cx, cy = Utilities.centroid_approximate(im, blur_radius=5, subframe_size=10)
        windowSize = 40

        average = (im.sum() - im[int(round(cy - windowSize)):int(round(cy + windowSize)),
                              int(round(cx - windowSize)):int(round(cx + windowSize))].sum()) / im.size
        self.bias_level = average
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

    def setup_AOI(self, half_width, half_height):
        half_width = int(round(half_width))
        half_height = int(round(half_height))
        self.fgs_cam.reset_AOI()
        image = self.get_frame() - self.bias_level
        blur_radius = 5
        subframe_size = self.fwhm_x * 1.5
        centroid = self.find_centroid_blind(blur_radius, subframe_size)
        x_off = int(round(centroid[0])) - half_width
        y_off = int(round(centroid[1])) - half_height
        self.fgs_cam.set_AOI_size(half_width * 2 + 1, half_height * 2 + 1)
        self.fgs_cam.set_AOI_position(x_off, y_off)
        return x_off, y_off

    def pid_start(self, target_x, target_y, repetitions):
        set_v = [50.0, 50.0, 50.0]
        self.set_voltages(*set_v)
        time.sleep(1)

        x_off, y_off = self.setup_AOI(self.fwhm_x * 3, self.fwhm_x * 3)
        width, height = self.fgs_cam.get_AOI_size()
        frame_center_x = int(round(width / 2.0)) + x_off
        frame_center_y = int(round(height / 2.0)) + y_off

        # If centroid is not in the center of the subframe, adjust subframe
        # but only do it after the distance is greater that these thresholds:
        min_mov_x = self.fwhm_x / 3.0
        min_mov_x = 3 if min_mov_x < 3 else min_mov_x
        min_mov_y = self.fwhm_y / 3.0
        min_mov_y = 3 if min_mov_y < 3 else min_mov_y

        pid_x = PID(0.5, 0.0, 0.0, setpoint=target_x)
        pid_y = PID(0.5, 0.0, 0.0, setpoint=target_y)
        pid_x.output_limits = (-self.fwhm_x * 1.5, self.fwhm_x * 1.5) # in pixels
        pid_y.output_limits = (-self.fwhm_x * 1.5, self.fwhm_x * 1.5)

        errs = np.zeros((repetitions, 2))
        pos = np.zeros((repetitions, 2))
        t = np.zeros(repetitions)

        # self.fgs_cam.set_AOI_size(40, 40)
        # self.fgs_cam.set_AOI_position(672, 566)
        # self.fgs_cam.set_clock(self.fgs_cam.get_clock_limits()[1])
        # self.fgs_cam.set_exposure(5)
        # self.fgs_cam.set_framerate(100)
        im = self.get_frame()
        tot_weight = im.sum()
        plt.figure()
        plt.imshow(im)
        plt.show()
        time.sleep(1)

        for i in range(0, repetitions):
            img_fgs = self.fgs_cam.acquire() - self.bias_level
            if img_fgs.sum() < (tot_weight / 2.0):
                #print("Low weight")
                x_off, y_off = self.setup_AOI(self.fwhm_x * 3, self.fwhm_x * 3)
                width, height = self.fgs_cam.get_AOI_size()
                frame_center_x = int(round(width / 2.0)) + x_off
                frame_center_y = int(round(height / 2.0)) + y_off
                img_fgs = self.fgs_cam.acquire() - self.bias_level

            cx, cy = self.find_centroid(img_fgs, x_offset=x_off, y_offset=y_off)

            # TODO: Change to a fast move AOI, and check it works
            if (abs(frame_center_x - cx) > min_mov_x) or (abs(frame_center_y - cy) > min_mov_y):
                #print("off-set! Moving AOI")
                x_off = int(round(cx)) - int(round(width / 2.0))
                y_off = int(round(cy)) - int(round(height / 2.0))
                self.fgs_cam.set_AOI_position_fast(x_off, y_off)
                frame_center_x = int(round(width / 2.0)) + x_off
                frame_center_y = int(round(height / 2.0)) + y_off
                img_fgs = self.fgs_cam.acquire() - self.bias_level
                #tot_weight = im.sum()
                cx, cy = self.find_centroid(img_fgs, x_offset=x_off, y_offset=y_off)

            errs[i, :] = [cx - target_x, cy - target_y]
            pos[i, :] = [cx, cy]
            t[i] = time.time()
            x_control = pid_x(cx)
            y_control = pid_y(cy)
            delta_voltages = self.calculate_piezo_offset(x_control, y_control)
            set_v[0] = set_v[0] + delta_voltages[0]
            set_v[1] = set_v[1] + delta_voltages[1]
            set_v[2] = set_v[2] + delta_voltages[2]
            self.set_voltages(*set_v)

        return pos, errs, t-t.min()

    def test_speed(self):
        self.fgs_cam.set_AOI_size(20, 20)
        self.fgs_cam.set_AOI_position(500, 500)
        self.fgs_cam.set_clock(self.fgs_cam.get_clock_limits()[1])
        self.fgs_cam.set_exposure(5)
        self.fgs_cam.set_framerate(1000/5)

        print("FPS: %f" % self.fgs_cam.get_framerate())
        print("clock: %f" % self.fgs_cam.get_clock())
        print("exp: %f" % self.fgs_cam.get_exposure())
        #self.fgs_cam.set_framerate(200)

        # test FPS with AOI:
        t_i = time.time()

        for n in range(0, 100):
            img_fgs = self.fgs_cam.acquire()
            self.calculate_piezo_offset(0, 0)
            #self.set_voltages(50, 50, 50)

        t_e = time.time()
        print("Time needed: %f" % (t_e - t_i))
        print("FPS: %f" % (100 / (t_e - t_i)))

        self.fgs_cam.reset_AOI()
        return img_fgs

    def find_centroid_blind(self, blur_radius=5, subframe_size=10):
        self.fgs_cam.reset_AOI()
        self.get_bias_level()
        img_fgs = self.fgs_cam.acquire() - self.bias_level
        cx, cy = Utilities.initial_centroid(img_fgs, blur_radius, subframe_size)
        return [cx, cy]

    @staticmethod
    def find_centroid(image, x_offset=0, y_offset=0, window_size=10):
        cx, cy, _, _ = Utilities.centroid_iterative(image, x_offset, y_offset, convergence_limit=0.5, max_iterations=10, window_size=window_size)
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
                img_fgs = self.fgs_cam.acquire() - self.bias_level
                cx_temp, cy_temp, _, _ = Utilities.centroid_iterative(img_fgs)
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
