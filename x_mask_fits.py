import numpy as np
from astropy.io import fits

#make a 6000, 6000, 1, 1 of ones, with a 100, 100, 1, 1 square of zeros in the middle
data = np.ones((1, 1, 6000, 6000))

data[0, 0, 2950:3050, 2950:3050] = 0

# save that to a fits file
hdu = fits.PrimaryHDU(data)
hdu.writeto('mask.fits', overwrite=True)

print(hdu[0].data.shape)