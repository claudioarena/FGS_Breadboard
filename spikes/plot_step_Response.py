import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import shelve
from matplotlib.ticker import EngFormatter
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

save_pics_pdf = True

# steps response 1
filename = 'data/march2020/10-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_step_1 = pos
time_step_1 = time
total_light_step_1 = total_light
slit_light_step_1 = slit_light

# steps response 2
filename = 'data/march2020/11-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_step_2 = pos
time_step_2 = time
total_light_step_2 = total_light
slit_light_step_2 = slit_light

#scatter plot positions
#Cut the first 3.5 s of data
first_x = [ n for n,i in enumerate(time_step_1) if i>1][0]

cx_step_1 = np.average(pos_step_1[first_x:first_x+100, 0])
cy_step_1 = np.average(pos_step_1[first_x:first_x+100, 1])
cx_step_2 = np.average(pos_step_2[first_x:first_x+100, 0])
cy_step_2 = np.average(pos_step_2[first_x:first_x+100, 1])

time_step_2 = time_step_2 - 5
first_x = [ n for n,i in enumerate(time_step_2) if i>0][0]
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_step_2[first_x:, 0])
cy_PID = np.average(pos_step_2[first_x:, 1])
axs.scatter(time_step_2[first_x:], pos_step_2[first_x:, 0] - cx_step_2, marker="+", s=10, c='red', label='X axis')
axs.scatter(time_step_2[first_x:], pos_step_2[first_x:, 1] - cy_step_2, marker="+", s=10, c='blue', label='Y axis')
axs.set_xlim([0,15.5])
axs.set_ylim([-1,11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

first_x = [ n for n,i in enumerate(time_step_2) if i>2.5][0]
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_step_2[first_x:, 0])
cy_PID = np.average(pos_step_2[first_x:, 1])
axs.scatter(time_step_2[first_x:], pos_step_2[first_x:, 0] - cx_step_2, marker="+", s=10, c='red', label='X axis')
axs.scatter(time_step_2[first_x:], pos_step_2[first_x:, 1] - cy_step_2, marker="+", s=10, c='blue', label='Y axis')
axs.set_xlim([4.5,7.0])
axs.set_ylim([-1,11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot_zoomed.pdf', dpi=600, format='pdf')

plt.show()

