import shelve

filename = 'data/march2020/11-shelve.out'
my_shelf = shelve.open(filename, 'n') # 'n' for new

list = ['pos','time','slit_light','total_light','a!',"Pxx_den","errs","centroid","exp_time","f","fs","gain","gain_boost",
        "image","max_fps","x_off","y_off","axis_sense","axis_angle","axis_x_proj","axis_y_proj","cx_X", "cy_X", "v_range_X", "d_X",
        "cx_Y", "cy_Y", "v_range_Y", "d_Y","cx_Z", "cy_Z", "v_range_Z", "d_Z"]

#list_exclude = ['exit', 'CDLL.__init__.<locals>._FuncPtr',
#                'get_ipython', 'plt', 'sys', 'np','signal',
#                'quit','fgs_var', 'shelve']
for key in list:
#for key in dir():
    if key.startswith('_') or key is "quit" or key is "fgs_var" or key is "get_ipython" or \
            key is "plt" or key is "sys" or key is "np" or key is "signal" or key is "shelve" or key is "exit":
        continue

    try:
        #print(key)
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        ##
        print('ERROR shelving: {0}'.format(key))
    except Exception as e:
        print('Other Error')
        print(e)

my_shelf.close()

##Restore
import shelve

filename = 'data/march2020/10-shelve.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()