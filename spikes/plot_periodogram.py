import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import shelve
from matplotlib.ticker import EngFormatter
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import integrate

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
filename = 'data/march2020/3-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_noPID = pos
time_noPID = time
total_light_noPID = total_light

#PID file
filename = 'data/march2020/4-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_PID = pos
time_PID = time
total_light_PID = total_light

#No PID, no piezo file
filename = 'data/march2020/5-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
pos_NoPiezo = pos
time_NoPiezo = time
total_light_NoPiezo = total_light

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
first_x = [ n for n,i in enumerate(time_PID) if i>3.5 ][0]

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(pos_PID[first_x:, 0] - cx_PID, pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='red', label='PID Guiding On')
limit_n = -1.5
limit_p = 1.5
ticks_spacing = 0.5
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(x_ticks)
plt.yticks(x_ticks)
plt.xticks(rotation=90)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')

if save_pics_pdf:
    plt.savefig('scatter_PID_On_v2.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
#plt.subplot(2,1,1,constrained_layout=True)
#plt.title("X and Y centroid deviations")
cx_PID = np.average(pos_PID[first_x:, 0])
cy_PID = np.average(pos_PID[first_x:, 1])
axs.set_title('PID Guiding On')
axs.scatter(pos_PID[first_x:, 0] - cx_PID, pos_PID[first_x:, 1] - cy_PID, marker="+", s=10, c='red', label='PID Guiding On')
limit_n = -0.3
limit_p = 0.3
ticks_spacing = 0.1
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(x_ticks)
plt.yticks(x_ticks)
plt.xticks(rotation=90)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
print('PID Guiding On, RMS: x:{} pix y:{} pix'.format(np.std(pos_PID[first_x:, 0]), np.std(pos_PID[first_x:, 1])))
d_squared = (np.power((pos_PID[first_x:, 0] - cx_PID), 2) + np.power((pos_PID[first_x:, 1] - cy_PID), 2))
distance = np.sqrt(d_squared)
tot_stdev = np.sqrt(np.sum(d_squared) / len(d_squared))
print('PID Guiding On, RMS total: {} pixels'.format(tot_stdev))

if save_pics_pdf:
    plt.savefig('scatter_PID_On_zoomed_v2.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
cx_noPID = np.average(pos_noPID[first_x:, 0])
cy_noPID = np.average(pos_noPID[first_x:, 1])
axs.set_title('PID Guiding Off')
axs.scatter(pos_noPID[first_x:, 0] - cx_noPID, pos_noPID[first_x:, 1] - cy_noPID, marker="+", s=10, c='blue', label='PID Guiding Off')
limit_n = -1.5
limit_p = 1.5
ticks_spacing = 0.5
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(x_ticks)
plt.yticks(x_ticks)
plt.xticks(rotation=90)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
print('PID Guiding Off, RMS: x:{} pix y:{} pix'.format(np.std(pos_noPID[first_x:, 0]), np.std(pos_noPID[first_x:, 1])))
d_squared = (np.power((pos_noPID[first_x:, 0] - cx_noPID), 2) + np.power((pos_noPID[first_x:, 1] - cy_noPID), 2))
distance = np.sqrt(d_squared)
tot_stdev = np.sqrt(np.sum(d_squared) / len(d_squared))
print('PID Guiding Off, RMS total: {} pixels'.format(tot_stdev))
if save_pics_pdf:
    plt.savefig('scatter_PID_Off_v2.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3/2, 6.3/2))
cx_NoPiezo = np.average(pos_NoPiezo[first_x:, 0])
cy_NoPiezo = np.average(pos_NoPiezo[first_x:, 1])
axs.set_title('No Piezo connection')
axs.scatter(pos_NoPiezo[first_x:, 0] - cx_NoPiezo, pos_NoPiezo[first_x:, 1] - cy_NoPiezo, marker="+", s=10, c='blue', label='Piezo disconnected')
limit_n = -1.5
limit_p = 1.5
ticks_spacing = 0.5
axs.set_xlim([limit_n, limit_p])
axs.set_ylim([limit_n, limit_p])
x_ticks = np.linspace(limit_n, limit_p, round((limit_p-limit_n)/ticks_spacing+1))
plt.xticks(x_ticks)
plt.yticks(x_ticks)
plt.xticks(rotation=90)
axs.set_xlabel('X (pixels)')
axs.set_ylabel('Y (pixels)')
print('no piezo, RMS: x:{} pix y:{} pix'.format(np.std(pos_NoPiezo[first_x:, 0]), np.std(pos_NoPiezo[first_x:, 1])))
d_squared = (np.power((pos_NoPiezo[first_x:, 0] - cx_NoPiezo), 2) + np.power((pos_NoPiezo[first_x:, 1] - cy_NoPiezo), 2))
distance = np.sqrt(d_squared)
tot_stdev = np.sqrt(np.sum(d_squared) / len(d_squared))
print('no piezo, RMS total: {} pixels'.format(tot_stdev))
if save_pics_pdf:
    plt.savefig('scatter_PID_Off_disconnected_v2.pdf', dpi=600, format='pdf')

#power spectrums
fs_PID = 1/ ( (time_PID[500] - time_PID[100]) / 400 )
fs_noPID = 1/ ( (time_noPID[500] - time_noPID[100]) / 400 )
fs_NoPiezo = 1/ ( (time_NoPiezo[500] - time_NoPiezo[100]) / 400 )

##diversion from linear time: +- 0.002 sec
f_x_PID, Px_den_PID = signal.periodogram(pos_PID[first_x:, 0], fs_PID)
f_y_PID, Py_den_PID = signal.periodogram(pos_PID[first_x:, 1], fs_PID)
f_x_noPID, Px_den_noPID = signal.periodogram(pos_noPID[first_x:, 0], fs_noPID)
f_y_noPID, Py_den_noPID = signal.periodogram(pos_noPID[first_x:, 1], fs_noPID)
f_x_NoPiezo, Px_den_NoPiezo = signal.periodogram(pos_NoPiezo[first_x:, 0], fs_NoPiezo)
f_y_NoPiezo, Py_den_NoPiezo = signal.periodogram(pos_NoPiezo[first_x:, 1], fs_NoPiezo)

#plt.figure()
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
#plt.subplot(2,1,1,constrained_layout=True)
axs.set_title('PID Guiding On')
axs.loglog(f_x_noPID[1:], Px_den_noPID[1:], c='darkgray', linewidth=0.6)
axs.loglog(f_y_noPID[1:], Py_den_noPID[1:], c='darkgray', linewidth=0.6)
axs.loglog(f_x_PID[1:], Px_den_PID[1:], c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_PID[1:], Py_den_PID[1:], c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[0].set_ylim([1e-8, 1e-1])
axs.set_ylim([1e-10, 1e3])
formatter0 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter0)
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.set_xlabel('Frequency')
axs.set_ylabel('PSD [${Pixels_{rms}}^2 / Hz$]')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('psd_PID_On_v3.pgf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))

#plt.subplot(2,1,2,constrained_layout=True)
axs.set_title('PID Guiding Off')
axs.loglog(f_x_noPID[1:], Px_den_noPID[1:], c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_noPID[1:], Py_den_noPID[1:], c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[1].set_ylim([1e-10, 1e3])
axs.set_ylim([1e-10, 1e3])
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
    plt.savefig('psd_PID_Off_v2.pgf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))

#plt.subplot(3,1,2,constrained_layout=True)
axs.set_title('PID Guiding Off, No Piezo Connection')
axs.loglog(f_x_NoPiezo[1:], Px_den_NoPiezo[1:], c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_NoPiezo[1:], Py_den_NoPiezo[1:], c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[1].set_ylim([1e-10, 1e3])
axs.set_ylim([1e-10, 1e3])
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
    plt.savefig('psd_PID_Off_disconnected_v2.pgf')


######## Cumulative Power Spectrum #########
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
#plt.subplot(2,1,1,constrained_layout=True)
axs.set_title('PID Guiding On')
f_x_interval = [f_x_PID[i+1]-f_x_PID[i] for i in range(len(f_x_PID)-1)]
f_y_interval = [f_y_PID[i+1]-f_y_PID[i] for i in range(len(f_y_PID)-1)]
x_int = np.sqrt(np.cumsum(Px_den_PID[:-1] * f_x_interval))
y_int = np.sqrt(np.cumsum(Py_den_PID[:-1] * f_y_interval))
#x_int = np.sqrt(integrate.cumtrapz(Px_den_PID[1:], f_x_PID[1:], initial=0))
#y_int = np.sqrt(integrate.cumtrapz(Py_den_PID[1:], f_y_PID[1:], initial=0))
axs.loglog(f_x_PID[1:], x_int, c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_PID[1:], y_int, c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[0].set_ylim([1e-8, 1e-1])
axs.set_ylim([1e-10, 1e3])
formatter0 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter0)
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.set_xlabel('Frequency')
axs.set_ylabel('CSP [${Pixels_{rms}}$]')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('csp_PID_On.pgf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))

#plt.subplot(2,1,2,constrained_layout=True)
axs.set_title('PID Guiding Off')
f_x_interval = [f_x_noPID[i+1]-f_x_noPID[i] for i in range(len(f_x_noPID)-1)]
f_y_interval = [f_y_noPID[i+1]-f_y_noPID[i] for i in range(len(f_y_noPID)-1)]
x_int = np.sqrt(np.cumsum(Px_den_noPID[:-1] * f_x_interval))
y_int = np.sqrt(np.cumsum(Py_den_noPID[:-1] * f_y_interval))
#x_int = np.sqrt(integrate.cumtrapz(Px_den_noPID[1:], f_x_noPID[1:], initial=0))
#y_int = np.sqrt(integrate.cumtrapz(Py_den_noPID[1:], f_y_noPID[1:], initial=0))
axs.loglog(f_x_noPID[1:], x_int, c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_noPID[1:], y_int, c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[1].set_ylim([1e-10, 1e3])
axs.set_ylim([1e-10, 1e3])
formatter1 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter1)
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.set_xlabel('Frequency')
axs.set_ylabel('CSP [${Pixels_{rms}}$]')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('csd_PID_Off.pgf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))

#plt.subplot(3,1,2,constrained_layout=True)
axs.set_title('PID Guiding Off, No Piezo Connection')
f_x_interval = [f_x_NoPiezo[i+1]-f_x_NoPiezo[i] for i in range(len(f_x_NoPiezo)-1)]
f_y_interval = [f_y_NoPiezo[i+1]-f_y_NoPiezo[i] for i in range(len(f_y_NoPiezo)-1)]
x_int = np.sqrt(np.cumsum(Px_den_NoPiezo[:-1] * f_x_interval))
y_int = np.sqrt(np.cumsum(Py_den_NoPiezo[:-1] * f_y_interval))
#x_int = np.sqrt(integrate.cumtrapz(Px_den_NoPiezo[1:], f_x_NoPiezo[1:], initial=0))
#y_int = np.sqrt(integrate.cumtrapz(Py_den_NoPiezo[1:], f_y_NoPiezo[1:], initial=0))
axs.loglog(f_x_NoPiezo[1:], x_int, c='red', label='X axis', linewidth=0.6)
axs.loglog(f_y_NoPiezo[1:], y_int, c='blue', label='Y axis', linewidth=0.6)
axs.set_xlim([1e-3, 4*1e1])
#axs[1].set_ylim([1e-10, 1e3])
axs.set_ylim([1e-10, 1e3])
formatter1 = EngFormatter(unit='Hz')
axs.xaxis.set_major_formatter(formatter1)
axs.yaxis.grid(b=True, which='major', color='#888888', linestyle='--')
#axs.yaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
#axs.xaxis.grid(b=True, which='minor', color='#888888', linestyle='--')
axs.xaxis.grid(b=True, which='major', color='#888888', linestyle='--')
axs.set_xlabel('Frequency')
axs.set_ylabel('CSP [${Pixels_{rms}}$]')
axs.legend(loc='best',)

if save_pic:
    plt.savefig('csp_PID_Off_disconnected.pgf')
