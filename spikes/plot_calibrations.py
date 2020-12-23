import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import shelve
from matplotlib.ticker import EngFormatter
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

save_pics_pdf = True

# cal 1
filename = 'data/march2020/1-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_X, cx_X, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_X, cy_X, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_Y, cx_Y, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_Y, cy_Y, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_Z, cx_Z, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(v_range_Z, cy_Z, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_X, cy_X, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_Y, cy_Y, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_Z, cy_Z, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

# cal 2
filename = 'data/march2020/2-shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_X, cy_X, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_Y, cy_Y, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')

fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/2))
axs.scatter(cx_Z, cy_Z, marker="+", s=10, c='red', label='X axis')
#axs.set_xlim([0, 15.5])
#axs.set_ylim([-1, 11])
axs.set_xlabel('Time (seconds)')
axs.set_ylabel('Pixels')
axs.legend(loc='best',)
if save_pics_pdf:
    plt.savefig('step_time_plot.pdf', dpi=600, format='pdf')
