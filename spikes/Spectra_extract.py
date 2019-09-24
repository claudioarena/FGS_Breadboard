import pydis
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

apwidth = 10
skysep = 25
skywidth = 15


def showExctractionArea():
    xbins = np.arange(img_spectra.shape[1])

    plt.figure()
    plt.imshow(np.log10(img_spectra), origin='lower',aspect='auto',cmap=cm.Greys_r)

    # the trace
    plt.plot(xbins, trace,'b',lw=1)

    # the aperture
    plt.plot(xbins, trace-apwidth,'r',lw=1)
    plt.plot(xbins, trace+apwidth,'r',lw=1)

    # the sky regions
    plt.plot(xbins, trace-apwidth-skysep,'g',lw=1)
    plt.plot(xbins, trace-apwidth-skysep-skywidth,'g',lw=1)
    plt.plot(xbins, trace+apwidth+skysep,'g',lw=1)
    plt.plot(xbins, trace+apwidth+skysep+skywidth,'g',lw=1)

    plt.title('(with trace, aperture, and sky regions)')
    plt.draw()

# Read the data.
img_spectra = fits.getdata(filename="Spectra_image.fit")

#Bias calibrate. Might not be needed.
#bias = pydis.biascombine('rbias.lis', trim=True)
#data = (img_spectra - bias)

#Show data
plt.figure()
plt.imshow(np.log10(img_spectra), origin='lower', aspect='auto', cmap=cm.Greys_r)
plt.show()

#Wavelength cal
HeNeAr_file = 'Spectra_image.fit'
dataRegion = [1, img_spectra.shape[0]+1, 532, 572]
#wfit = pydis.HeNeAr_fit(HeNeAr_file, trim=False, interac=True, mode='poly', fit_order=5, disp_approx=0.1, wcen_approx=600.0, datasec=dataRegion)

# trace the science image
trace = pydis.ap_trace(img_spectra, nsteps=10, interac=False, display=True)
showExctractionArea()
#extract
ext_spec, sky, fluxerr = pydis.ap_extract(img_spectra, trace, apwidth=apwidth, skysep=skysep, skywidth=skywidth, skydeg=0)
flux_red = (ext_spec - sky)  # the reduced object

#normalise
flux_red = (flux_red - np.min(flux_red))
flux_red = (flux_red / np.max(flux_red))
#Wavelenght calibrate
#wfinal = pydis.mapwavelength(trace, wfit, mode='poly')

#From rough wav cal: y = -3E-05x2 + 0.1283x + 524.46R² = 0.995
x = np.arange(1,img_spectra.shape[1]+1)
wav = -0.00006*x*x + 0.155*x + 519
#wav = 0.0876*x + 529.85

#Display result
plt.figure()
plt.plot(wav,flux_red)
#plt.errorbar(flux_red, yerr=fluxerr)
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title("Spectra")
#plot within percentile limits
plt.ylim(0,1)
plt.show()