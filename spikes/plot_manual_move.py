import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import shelve
from matplotlib.ticker import EngFormatter
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import ticker

save_pic = False
save_pics_pdf = False

if save_pic:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

# No PID file
filename = 'data/march2020/9-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_noPID = pos
time_noPID = time
total_light_noPID = total_light
slit_light_noPID = slit_light

#PID file
filename = 'data/march2020/8-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_PID = pos
time_PID = time
total_light_PID = total_light
slit_light_PID = slit_light

# Timeline positions
#plt.figure()
#plt.scatter(time_PID, pos_PID[:, 0], c='coral')
#plt.figure()
#plt.scatter(time_PID, pos_PID[:, 1], c='lightblue')

#plt.figure()
#plt.scatter(time, total_light, c='lightblue')
#plt.figure()
#plt.scatter(time, slit_light, c='lightblue')
#plt.show()

#scatter plot positions
#Cut the first 3.5 s of data
first_x = [ n for n,i in enumerate(time_PID) if i>0.7][0]

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(pos_PID[first_x:, 0] - cx_PID, pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='red', label='PID Guiding On')
limit_n = -30
limit_p = 30
ticks_spacing = 10
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(rotation=90)
plt.xticks(x_ticks)
plt.yticks(x_ticks)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
print('pid on, RMS: x:{} pix y:{} pix'.format(np.std(pos_PID[first_x:, 0]), np.std(pos_PID[first_x:, 1])))
d_squared = (np.power((pos_PID[first_x:, 0] - cx_PID), 2) + np.power((pos_PID[first_x:, 1] - cy_PID), 2))
distance = np.sqrt(d_squared)
tot_stdev = np.sqrt(np.sum(d_squared) / len(d_squared))
print('pid on, RMS total: {} pixels'.format(tot_stdev))

if save_pics_pdf:
    plt.savefig('man_move_scatter_PID_On.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(pos_PID[first_x:, 0] - cx_PID, pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='red', label='PID Guiding On')
limit_n = -1
limit_p = 1
ticks_spacing = 0.5
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(rotation=90)
plt.xticks(x_ticks)
plt.yticks(x_ticks)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')

if save_pics_pdf:
    plt.savefig('man_move_scatter_PID_On_zoomed.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
cx_noPID = np.average(pos_noPID[first_x:, 0])
cy_noPID = np.average(pos_noPID[first_x:, 1])
axs.set_title('PID Guiding Off')
axs.scatter(pos_noPID[first_x:, 0] - cx_noPID, pos_noPID[first_x:, 1] - cy_noPID, marker="+", s=10, c='blue', label='PID Guiding Off')
limit_n = -30
limit_p = 30
ticks_spacing = 10
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(rotation=90)
plt.xticks(x_ticks)
plt.yticks(x_ticks)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
print('pid off, RMS: x:{} pix y:{} pix'.format(np.std(pos_noPID[first_x:, 0]), np.std(pos_noPID[first_x:, 1])))
d_squared = (np.power((pos_noPID[first_x:, 0] - cx_noPID), 2) + np.power((pos_noPID[first_x:, 1] - cy_noPID), 2))
distance = np.sqrt(d_squared)
tot_stdev = np.sqrt(np.sum(d_squared) / len(d_squared))
print('pid off, RMS total: {} pixels'.format(tot_stdev))

if save_pics_pdf:
    plt.savefig('man_move_scatter_PID_Off.pdf', dpi=600, format='pdf')

#power spectrums
fs_PID = 1/ ( (time_PID[500] - time_PID[100]) / 400 )
fs_noPID = 1/ ( (time_noPID[500] - time_noPID[100]) / 400 )

##diversion from linear time: +- 0.002 sec
f_x_PID, Px_den_PID = signal.periodogram(pos_PID[first_x:, 0], fs_PID)
f_y_PID, Py_den_PID = signal.periodogram(pos_PID[first_x:, 1], fs_PID)
f_x_noPID, Px_den_noPID = signal.periodogram(pos_noPID[first_x:, 0], fs_noPID)
f_y_noPID, Py_den_noPID = signal.periodogram(pos_noPID[first_x:, 1], fs_noPID)

#plt.figure()
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
#plt.subplot(2,1,1,constrained_layout=True)
axs.set_title('PID Guiding On')
axs.loglog(f_x_noPID[1:], Px_den_noPID[1:], c='darkgray', linewidth=0.6)
axs.loglog(f_y_noPID[1:], Py_den_noPID[1:], c='darkgray', linewidth=0.6)
axs.loglog(f_x_PID[1:], Px_den_PID[1:], c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_PID[1:], Py_den_PID[1:], c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([30*1e-3, 15*1e1])
axs.set_ylim([1e-8, 1e3])
formatter0 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter0)
axs.set_xlabel('Frequency')
axs.set_ylabel('PSD [${Pixels_{rms}}^2 / Hz$]')
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('man_move_psd_PID_On_v3.pgf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))

#plt.subplot(2,1,2,constrained_layout=True)
axs.set_title('PID Guiding Off')
axs.loglog(f_x_noPID[1:], Px_den_noPID[1:], c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_noPID[1:], Py_den_noPID[1:], c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([30*1e-3, 15*1e1])
#axs[1].set_ylim([1e-10, 1e3])
axs.set_ylim([1e-8, 1e3])
formatter1 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter1)
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.set_xlabel('Frequency')
axs.set_ylabel('PSD [${Pixels_{rms}}^2 / Hz$]')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('man_move_psd_PID_Off_v2.pgf')


first_x = [ n for n,i in enumerate(time_noPID) if i>0.7][0]
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_noPID = np.average(pos_noPID[first_x:, 0])
cy_noPID = np.average(pos_noPID[first_x:, 1]) + 1 #man adjust position to centered
axs.set_title('PID Guiding Off')
axs.scatter(time_noPID[first_x:], pos_noPID[first_x:, 0] - cx_noPID, marker="+", s=10, c='red', label='X axis')
axs.scatter(time_noPID[first_x:], pos_noPID[first_x:, 1] - cy_noPID, marker="+", s=10, c='blue', label='Y axis')
axs.set_xlim([0,30])
axs.set_ylim([-15,15])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

if save_pics_pdf:
    plt.savefig('man_move_time_plot_PID_Off_v2.pdf', dpi=600, format='pdf')

first_x = [ n for n,i in enumerate(time_PID) if i>0.7][0]
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(time_PID[first_x:], pos_PID[first_x:, 0] - cx_PID, marker="+", s=10, c='red', label='X axis')
axs.scatter(time_PID[first_x:], pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='blue', label='Y axis')
axs.set_xlim([0,30])
axs.set_ylim([-15,15])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('man_move_time_plot_PID_On_v2.pdf', dpi=600, format='pdf')

first_x = [ n for n,i in enumerate(time_PID) if i>0.7][0]
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(time_PID[first_x:], pos_PID[first_x:, 0] - cx_PID, marker="+", s=10, c='red', label='X axis')
axs.scatter(time_PID[first_x:], pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='blue', label='Y axis')
axs.set_xlim([0,30])
axs.set_ylim([-1,1])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('man_move_time_plot_PID_On_zoomed_v2.pdf', dpi=600, format='pdf')

plt.show()

