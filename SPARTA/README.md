# SPARTA (Spectral Processing for Analysis, Reduction and Two-source Astrometry)

A program for the processing of spectro-astrometric observations. The workflow is divided into three stages, each executed in a separate graphical window.

## Running
To ensure the program works correctly, it is recommended to use a virtual environment to avoid dependency conflicts.

1. Download the project files to your working directory.
2. Create and activate a virtual environment.
3. Install the required dependencies:

```bash
   pip install -r requirements.txt
```
4. Run the graphical interface:

```bash
   python SPARTA_main.py
```

The program works with FITS files. It is expected that the data array and the header are located in the primary (zeroth) extension of the file. If your file structure differs, they must be converted beforehand.

---

## Stage 1: Preprocessing

<div align="center">
  <img src="../img/SPARTA/SPARTA_preprocessing.png" width="100%">
</div>

At this stage, spectro-astrometric frames obtained at two position angles are loaded (`Load PA1` and `Load PA2`), the processing area is configured, and the primary signal is extracted.

### FITS Settings
If the FITS headers do not contain the required keywords, replace the `FITS` value with numerical parameters:
* **NAXIS1 / NAXIS2** — number of pixels along the horizontal and vertical axes.
* **CRVAL1** — starting wavelength of the frame.
* **CDELT1** — wavelength step per pixel.

### Parameters

| Parameter | Description |
|---|---|
| **Y_est / area** | Position of the spectrum center and its margins (along the Y-axis). Displayed as red dashed lines on the frame plot. They can be moved with the mouse (the center will adjust automatically). |
| **wl_start / wl_end** | The wavelength range under investigation (bounded by blue dashed lines, can be moved with the mouse). |
| **vmin / vmax** | Brightness range for displaying the frame (does not affect calculations). |
| **window_size** | Smoothing window size for the rolling mean method to remove trends from the spectra before combining them. |
| **med_sm** | Median smoothing parameter for the plots. |
| **ax_y_sigma** | Y-axis display scale for the bottom plots. |
| **proc_left / proc_right** | Cropping the frame edges along the X-axis (purple lines) to exclude edge noise. |

### Controls
In the **Options** block, you can enable/disable background removal (`rm_bg`), the use of wavelength boundaries (`wl_borders`), and the display of error bars on the plots (`see_err`).

* **Plot Interaction:** *Right-clicking* on any plot (except the frame itself) places a green synchronizing line across all panels to easily compare spectral features.
* **Plots:** Top right — averaged spectra. Bottom right — centroid deviations for PA1 and PA2. Bottom left — spectro-astrometric signal (half-difference of these deviations).
* **Analyze Data:** Saves the results of the first stage and opens the window for the next step.

---

## Stage 2: Spectro-astrometric Processing

<div align="center">
  <img src="../img/SPARTA/SPARTA_processing.png" width="100%">
</div>

At this stage, individual spectral lines are analyzed to determine the true Y-coordinates of the sources. The algorithm subtracts the spatial profile of the "continuum" spectrum from the profile within the line for multiple points around the line center. The results (measurements of the center coordinate) are displayed on three color maps at the bottom (PA1, PA2, and their half-difference).

### Parameters

| Parameter | Description |
|---|---|
| **line_center** | Center of the investigated line. It can be entered manually or by *right-clicking* on the top plots. |
| **d_cont** | Maximum offset from the line center to calculate coordinates. |
| **prof_area** | Width of the profile processing area (taken from Stage 1 by default). |
| **indent** | Offset from the coordinate matching points of the line and continuum profiles. |
| **vmin_1,2 / vmax_1,2** | Percentage of deviation of the color map points from the mean value to filter results for PA1 and PA2. |

### Workflow for Stage 2:
1. Select a line (`line_center`) and click **Create Map**. Evaluate the result on the color maps.
2. Click **Write Row** to record the Y-coordinate in the table on the left.
3. In the table's `source` column, indicate which of the two sources this line belongs to (1 or 2).
4. Repeat the process for other lines of both sources.
5. Click **Find Separation** so the program calculates the general coordinates and the separation `sep` (d1, d2) between the sources based on the collected data.
6. Click **Separate Spectra** to proceed to the final step.

---

## Stage 3: Spectral Separation

<div align="center">
  <img src="../img/SPARTA/SPARTA_separation.png" width="60%">
</div>

The final window separates the combined spectrum into two independent components using the `d1` and `d2` coordinates found in the previous stage.

* **Parameters:** You can manually change the `d1` and `d2` values to investigate their effect on the final result. The `med_sm` parameter applies median smoothing to the final plots.
* **Recount:** Recalculates the spectra using the newly entered d1/d2 values.
* **Set Defaults:** Restores the `d1` and `d2` values calculated in Stage 2.
* **Save Results:** Saves the final separated spectra to a file.
