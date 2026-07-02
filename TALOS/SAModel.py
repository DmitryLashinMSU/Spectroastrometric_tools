NAME = 'SAModel'

import numpy as np
import random


'''------------------------------------------Calculating the observed PSF--------------------------------------------'''


def PSF_calc(SLIT, SEEING, SEP, pix_size):

    M = 30  # Image scale-up

    # Image parameters
    width = 200 * M
    height = 150 * M

    # The real width of the slit
    slit_arcsec = SLIT

    # Atmospheric PSF parameters
    sigma = 5 * M  # Dispersion for an unscaled image
    seeing = SEEING  # Size of atmospheric PSF (arcsec)

    # Parameters of a subregion containing a central maximum
    region_width = 40 * M
    region_height = 40 * M

    # Parameters for separating objects along the slit
    sep = SEP  # Spectrum 2 offset relative to spectrum 1 (arcsec)

    # Image Parameters
    center_x = width // 2
    center_y = height // 2

    # Size and seeing for an unscaled image
    size_unscaled = 3 * sigma
    seeing_unscaled = 2 * sigma * np.sqrt(2 * np.log(2))

    # Slit parameters in the model image
    slit_width_image = int(slit_arcsec * seeing_unscaled / seeing)
    slit_position = width // 2

    # Parameters for separating objects along the slit
    offset_pix = sep * seeing_unscaled / seeing  # Spectrum 2 offset relative to spectrum 1 on frame (pix)

    image = np.zeros((height, width))
    # Array for the Gaussian function
    x, y = np.indices((height, width))
    gaussian = np.exp(-((x - center_y) ** 2 + (y - center_x) ** 2) / (2 * (sigma ** 2)))
    image += gaussian

    # Creating a mask for PSF
    mask = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            if slit_position - slit_width_image // 2 <= x <= slit_position + slit_width_image // 2:
                mask[y, x] = 1

    # PSF calculation
    diffraction_pattern = image * mask


    # Calculating coordinates of subdomain corners
    x_start = center_x - region_width // 2
    x_end = center_x + region_width // 2
    y_start = center_y - region_height // 2
    y_end = center_y + region_height // 2

    # Extracting a subregion of a PSF image
    region = np.abs(diffraction_pattern)[y_start:y_end, x_start:x_end]
    region_shifted = np.abs(diffraction_pattern)[int(y_start + offset_pix):int(y_end + offset_pix), x_start:x_end]

    # Calculating the scale of the PSF region on the CCD
    y_pixels = int(size_unscaled * pix_size * seeing / M) * 2  # Length of the scalable area in pixels
    x_pixels = int(y_pixels * region_width / region_height)
    scale_x = region_width // x_pixels
    scale_y = region_height // y_pixels

    PSF = np.zeros((y_pixels, x_pixels))
    for i in range(y_pixels):
        for j in range(x_pixels):
            PSF[i, j] = region[i * scale_y: (i + 1) * scale_y, j * scale_x: (j + 1) * scale_x].mean()

    Summ = 0
    for y in range(y_pixels):
        for x in range(y_pixels):
            Summ += PSF[y, x]

    PSF = PSF / Summ  # Normalization

    PSF_shifted = np.zeros((y_pixels, x_pixels))
    for i in range(y_pixels):
        for j in range(x_pixels):
            PSF_shifted[i, j] = region_shifted[i * scale_y: (i + 1) * scale_y, j * scale_x: (j + 1) * scale_x].mean()

    Summ2 = 0
    for y in range(y_pixels):
        for x in range(y_pixels):
            Summ2 += PSF_shifted[y, x]

    PSF_shifted = PSF_shifted / Summ2  # Normalization

    return(PSF + 1e-6, PSF_shifted + 1e-6)  # 1e-6 - a regularizing additive


'''-----------------------------------------------Spectrum modeling-------------------------------------------------'''


# Calculating the continuum according to Planck's law
def planck_function(wavelength, T):
    h = 6.62607015e-34  # Planck's constant (J·s)
    c = 299792458  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (J/K)
    wavelength = wavelength * 1e-10  # Convert to meters
    return (2 * h * c ** 2 / wavelength ** 5) * (1 / (np.exp(h * c / (wavelength * k * T)) - 1))


def ExampleSpectrum(T, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                    sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                    MaxMag_emission, MaxMag_absorption, MaxADU):

    wavelength_max = CRVAL1 + image_size_X / CDELT1
    wavelengths = np.arange(CRVAL1, wavelength_max, CDELT1)

    continuum = planck_function(wavelengths, T)

    # Definition of emission lines
    lines_emission = []
    for t in range(N_emission):
      line_wavelength_emission = random.randint(CRVAL1, int(wavelength_max ))
      magnitude_emission = random.uniform(MinMag_emission, MaxMag_emission)  # Line magnitude (random)
      lines_emission.append((line_wavelength_emission, magnitude_emission))

    # Adding emission lines to the spectrum
    for line_wavelength, magnitude in lines_emission:
      gaussian = np.exp(-((wavelengths - line_wavelength) / sigma_emission) ** 2 / 2)
      continuum *= (1 + magnitude * gaussian)

    # Definition of absorption lines
    lines_absorption = []
    for t in range(N_absorption):
      line_wavelength_absorption = random.randint(CRVAL1, int(wavelength_max))
      magnitude_absorption = random.uniform(MinMag_absorption, MaxMag_absorption)  # Line magnitude (random)
      lines_absorption.append((line_wavelength_absorption, magnitude_absorption))

    # Adding absorption lines to the spectrum
    for line_wavelength, magnitude in lines_absorption:
      gaussian = np.exp(-((wavelengths - line_wavelength) / sigma_absorption) ** 2 / 2)
      continuum *= (1 - magnitude * gaussian)

    intensity = continuum / max(continuum) * MaxADU

    return(wavelengths, intensity)


'''-------------------------------------------------Frame modeling---------------------------------------------------'''


# Option without using known spectra
def Frame_model(slit, seeing, sep, pix_size, T1, T2, CRVAL1, image_size_X, image_size_Y, Y_est, CDELT1,
                N_emission, N_absorption, sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                MaxMag_emission, MaxMag_absorption, MaxADU1, MaxADU2, RN, BN):

    PSF_kernel_1, PSF_kernel_2 = PSF_calc(slit, seeing, sep, pix_size)

    PSF_width = len(PSF_kernel_1[0])
    PSF_heigth = len(PSF_kernel_1)


    Lambda1, Int1 = ExampleSpectrum(T1, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                                           sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                                           MaxMag_emission, MaxMag_absorption, MaxADU1)

    spectrum1 = np.column_stack((Lambda1, Int1))
    for k1 in range(image_size_X):
        spectrum1[k1, 0] = int(round((Lambda1[k1] - CRVAL1) / CDELT1)) + PSF_width / 2

    Lambda2, Int2 = ExampleSpectrum(T2, CRVAL1, image_size_X, CDELT1, N_emission, N_absorption,
                                           sigma_emission, sigma_absorption, MinMag_emission, MinMag_absorption,
                                           MaxMag_emission, MaxMag_absorption, MaxADU2)

    spectrum2 = np.column_stack((Lambda2, Int2))
    for k2 in range(image_size_X):
        spectrum2[k2, 0] = int(round((Lambda2[k2] - CRVAL1) / CDELT1)) + PSF_width / 2


    image1 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image1[int(i + Y_est - PSF_heigth / 2), j + x] += Int1[x] * PSF_kernel_1[i][j]

    image2 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image2[int(i + Y_est - PSF_heigth / 2), j + x] += Int2[x] * PSF_kernel_2[i][j]

    # Final image without noise
    image = image1 + image2
    image = image[:, int(PSF_width / 2): - int(PSF_width / 2)]

    # Final image with noise
    image = image + BN  # Adding background
    image = np.random.poisson(image)  # Adding Poisson noise
    # Adding readout noise
    for i in range(image_size_X):
        for j in range(image_size_Y):
            R = random.randint(0, RN)
            image[j, i] += R

    return(image, PSF_kernel_1, Lambda1, Int1, Lambda2, Int2)


# Option with using known spectra
def Frame_model_known_spec(slit, seeing, sep, pix_size, CRVAL1, image_size_X, image_size_Y, Y_est, CDELT1, RN, BN,
                           Lambda1, Int1, Lambda2, Int2):

    PSF_kernel_1, PSF_kernel_2 = PSF_calc(slit, seeing, sep, pix_size)

    PSF_width = len(PSF_kernel_1[0])
    PSF_heigth = len(PSF_kernel_1)

    spectrum1 = np.column_stack((Lambda1, Int1))
    for k1 in range(image_size_X):
        spectrum1[k1, 0] = int(round((Lambda1[k1] - CRVAL1) / CDELT1)) + PSF_width / 2


    spectrum2 = np.column_stack((Lambda2, Int2))
    for k2 in range(image_size_X):
        spectrum2[k2, 0] = int(round((Lambda2[k2] - CRVAL1) / CDELT1)) + PSF_width / 2


    image1 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image1[int(i + Y_est - PSF_heigth / 2), j + x] += Int1[x] * PSF_kernel_1[i][j]

    image2 = np.zeros((image_size_Y, image_size_X + PSF_width))
    for x in range(image_size_X):
        for i in range(PSF_heigth):
            for j in range(PSF_width):
                image2[int(i + Y_est - PSF_heigth / 2), j + x] += Int2[x] * PSF_kernel_2[i][j]

    # Final image without noise
    image = image1 + image2
    image = image[:, int(PSF_width / 2): - int(PSF_width / 2)]

    # Final image with noise
    image = image + BN  # Adding background
    image = np.random.poisson(image)  # Adding Poisson noise
    # Adding readout noise
    for i in range(image_size_X):
        for j in range(image_size_Y):
            R = random.randint(0, RN)
            image[j, i] += R

    return(image, PSF_kernel_1)
