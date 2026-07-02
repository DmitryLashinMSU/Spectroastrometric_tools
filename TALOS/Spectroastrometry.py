NAME = 'Spectroastrometry'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf


'''-------------------------------------------Functions for approximation--------------------------------------------'''


''' Approximation of a "positive" Gaussian '''


def G_fit(coord, I):

    # Initial parameter values
    P0 = np.zeros(5)
    P0[0] = np.sum(I)  # Intensity integral
    P0[2] = np.mean(coord)  # Assessment of the center position
    P0[1] = np.sum((coord - P0[2]) ** 2 * I) / P0[0]  # Standard deviation estimate
    P0[3] = 5  # Initial background
    m_est = np.sum((coord - P0[2]) * I) / np.sum((coord - P0[2]) ** 2)
    P0[4] = m_est

    # Fit using the least squares method
    result = least_squares(L, P0, args=(coord, I))
    P = result.x
    return P


# Loss function
def L(x, coord, I):
    return I - G(coord, x)


# Approximating function
def G(coord, P):
    return P[0] / 2 * (erf((coord - P[2]) / (P[1] * np.sqrt(2))) +
                       erf((P[2] - coord + 1) / (P[1] * np.sqrt(2)))) + P[3]  # + P[4] * coord


''' Approximation of a "negative" Gaussian '''


def neg_G_fit(coord, I):

  # Initial parameter values
  P0 = np.zeros(5)
  I = -1 * I
  P0[0] = -1 * np.sum(np.max(I) - I)  # # Absorption integral
  P0[2] = np.mean(coord)  # Assessment of the center position
  P0[1] = np.sqrt(np.sum((coord - P0[2]) ** 2 * (np.min(I) - I)) / np.sum((np.min(I) - I))) # Standard deviation estimate
  P0[3] = np.max(I)  # Initial background
  m_est = np.sum((coord - P0[2]) * (np.min(I) - I)) / np.sum((coord - P0[2]) ** 2)
  P0[4] = m_est

  # Fit using the least squares method
  result = least_squares(neg_L, P0, args=(coord, I))
  P = result.x
  return P


# Loss function
def neg_L(x, coord, I):
    return neg_G(coord, x) - I


# Approximating function
def neg_G(coord, P):
    return - P[0] / 2 * (erf((coord - P[2]) / (P[1] * np.sqrt(2))) +
                         erf((P[2] - coord + 1) / (P[1] * np.sqrt(2)))) - P[3]  # + P[4] * coord


'''----------------------------------Extraction of spectroastrometric information------------------------------------'''


''' Calculation of the observed offset and FWHM '''


def Center_search(CRVAL1, CDELT1, x_start, x_end, area, Y_est, image):
    # Spectral slice
    X0 = x_start  # Start pixel
    X1 = x_end  # End pixel
    X_count = 0  # A counter for filling arrays in a loop

    LAMBDA = np.zeros(X1 - X0)
    CENTER = np.zeros(X1 - X0)
    ERRORBAR = np.zeros(X1 - X0)
    SPEC = np.zeros(X1 - X0)
    FWHM = np.zeros(X1 - X0)


    while X_count < (X1 - X0):
      slice_I = np.zeros(2 * area)  # Spectrum slice at the specified X_count coordinate
      coord = np.zeros(2 * area)
      for j in range(2 * area):
        slice_I[j] = image[Y_est - area + j][x_start + X_count]
        coord[j] = Y_est - area + j
        SPEC[X_count] += image[Y_est - area + j][ x_start + X_count]

      # Applying approximation to the model:
      P = G_fit(coord, slice_I)

      FWHM[X_count] = (2 * np.sqrt(2 * np.log(2)) * P[1])
      SNR = P[0] / P[3]
      error = 0.6 * FWHM[X_count] / SNR
      LAMBDA[X_count] = CRVAL1 + (x_start + X_count) * CDELT1
      CENTER[X_count] = P[2]
      ERRORBAR[X_count] = error

      X_count += 1

    return LAMBDA, SPEC, CENTER, FWHM, ERRORBAR


''' Estimation of the actual displacement based on lines (1) '''


def LCS_auto(CRVAL1, CDELT1, image, LAMBDA, SPEC, x_start, x_end, x_cont, Y_est, area, Line_type):

    # Line approximation

    coord_line = LAMBDA[x_start:x_end]
    I_line = SPEC[x_start:x_end]

    P1 = G_fit(coord_line, I_line)

    if Line_type == 'absorption':
        P1 = neg_G_fit(coord_line, I_line)

    if Line_type == 'emission':
        P1 = G_fit(coord_line, I_line)

    # area - half-width of the area across the dispersion direction in which the spectrum is analyzed
    Coord_center = np.zeros(2 * area)
    I_center = np.zeros(2 * area)
    for j in range(2 * area):
        I_center[j] = image[Y_est - area + j][int((P1[2] - CRVAL1) / CDELT1)]
        Coord_center[j] = Y_est - area + j

    P2 = G_fit(Coord_center, I_center)

    # Continuum approximation

    Coord_cont = np.zeros(2 * area)
    I_cont = np.zeros(2 * area)

    for j in range(2 * area):
        I_cont[j] = image[Y_est - area + j][x_cont]
        Coord_cont[j] = Y_est - area + j

    P3 = G_fit(Coord_cont, I_cont)

    # Line model

    Line = np.abs(G(Coord_center, P2) - G(Coord_cont, P3))
    P4 = G_fit(Coord_center, Line)
    return P4, (P1[2] - CRVAL1) / CDELT1


''' Estimation of the actual displacement based on lines (2): the center of the line is specified manually'''


def LCS_target(image, x_line, x_cont, Y_est, area):

    # area - half-width of the area across the dispersion direction in which the spectrum is approximated
    Coord_center = np.zeros(2 * area)
    I_center = np.zeros(2 * area)
    for j in range(2 * area):
        I_center[j] = image[Y_est - area + j][x_line]
        Coord_center[j] = Y_est - area + j

    P2 = G_fit(Coord_center, I_center)

    # Continuum approximation

    Coord_cont = np.zeros(2 * area)
    I_cont = np.zeros(2 * area)

    for j in range(2 * area):
        I_cont[j] = image[Y_est - area + j][x_cont]
        Coord_cont[j] = Y_est - area + j

    P3 = G_fit(Coord_cont, I_cont)

    # Line model

    Line = np.abs(G(Coord_center, P2) - G(Coord_cont, P3))
    P4 = G_fit(Coord_center, Line)
    return P4, P3


'''-------------------------------------------------Frame processing-------------------------------------------------'''

''' Median filtering '''

def Median_images(images):
    shape = images[0].shape
    if not all(img.shape == shape for img in images):
      raise ValueError("All images must have the same size!")

    stacked_images = np.stack(images, axis=-1)

    median_image = np.median(stacked_images, axis=-1)

    return median_image


''' Averaging '''


def Average_images(images):
    shape = images[0].shape
    if not all(img.shape == shape for img in images):
      raise ValueError("All images must have the same size!")

    stacked_images = np.stack(images, axis=-1)

    average_image = np.mean(stacked_images, axis=-1)

    return average_image


''' Trend elimination using the rolling mean method '''


def Detrend(window, CENTER):

    CENTER_rolling = np.zeros_like(CENTER)
    rolling_mean = np.zeros_like(CENTER)

    for i in range(len(CENTER)):
        start_index = max(0, i - window // 2)
        end_index = min(len(CENTER), i + window // 2 + 1)

        rolling_mean[i] = np.mean(CENTER[start_index:end_index])

        CENTER_rolling[i] = CENTER[i] - rolling_mean[i]

    return CENTER_rolling, rolling_mean


''' Eliminating the power trend '''


def remove_polynomial_trend(data, n):
    x = np.arange(len(data))
    coefficients = np.polyfit(x, data, n)
    trend = np.polyval(coefficients, x)
    detrended_data = data - trend
    return detrended_data


'''---------------------------------------------------Data display---------------------------------------------------'''


''' 3D graph for the selected part of the frame '''


def show_3D(image, x_start, x_end, y_start, y_end):

    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)

    # Area selection
    x_min, x_max = x_start, x_end
    y_min, y_max = y_start, y_end

    x_mask = (x >= x_min) & (x <= x_max)
    y_mask = (y >= y_min) & (y <= y_max)

    mask = x_mask & y_mask

    x_filtered = x[mask]
    y_filtered = y[mask]
    image_filtered = image[mask]

    # Conversion back to a 2D array for plot_surface (required for proper display)
    x_filtered = x_filtered.reshape(image[y_min:y_max+1, x_min:x_max+1].shape)
    y_filtered = y_filtered.reshape(image[y_min:y_max+1, x_min:x_max+1].shape)
    image_filtered = image[y_min:y_max+1, x_min:x_max+1]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_filtered, y_filtered, image_filtered / 1000, cmap='viridis')
    ax.set_xlabel('X, pix')
    ax.set_ylabel('Y, pix')
    ax.set_zlabel('I, ADU * 1e3')
    plt.show()
