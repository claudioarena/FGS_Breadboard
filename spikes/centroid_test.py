from FGS_Breadboard import fgs

fgs_var = fgs("COM5", 0)
try:
    fgs_var.setup_camera()
    n_average = 30
    v_start = 45
    v_end = 50
    v_step = 0.1
    pauseTime = 0.5
    fgs_var.calibrate(n_average, v_start, v_end, v_step, pauseTime)

finally:
    fgs_var.close()

#subframe = img_fgs[int(round(cx)-10):int(round(cx)+10), int(round(cy)-10):int(round(cy)+10)]
#pl.imshow(subframe)
#pl.scatter(x=(cy-int(round(cy)))+10, y=(cx-int(round(cx)))+10, c='r', s=40)

#pl.figure()
#pl.imshow(img_fgs, cmap='gray', vmin=img_fgs.min(), vmax=img_fgs.max())
#pl.show()


