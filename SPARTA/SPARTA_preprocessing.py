import os
import sys
import subprocess
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.ndimage import shift as nd_shift
from scipy.optimize import minimize
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# Auxiliary function for the rolling mean
def rolling_mean(data, window):
    if window < 1: return data
    return (np.convolve(data, np.ones(window), 'same') /
            np.convolve(np.ones_like(data), np.ones(window), 'same'))

# Class for interactive update
class InteractiveSpectroAstrometry:
    def __init__(self, app):
        self.app = app
        self.fig = app.fig
        self.ax_img = app.ax_img
        self.ax_spec = app.ax_spec
        self.ax_dev = app.ax_dev
        self.ax_hd = app.ax_hd

        self.h_lines = app.h_lines
        self.v_lines = app.v_lines

        self.spec_line1 = app.spec_line1
        self.spec_line2 = app.spec_line2
        self.dev_line1 = app.dev_line1
        self.dev_line2 = app.dev_line2
        self.hd_line = app.hd_line

        # Attributes for filling errors
        self.dev_fill1 = None
        self.dev_fill2 = None
        self.hd_fill = None

        # List for storing synchronous vertical lines
        self.sync_vlines = []

        self.master_1 = app.master_1
        self.master_2 = app.master_2

        self.dragging_line = None
        self.kind = None

        self.cids = [
            self.fig.canvas.mpl_connect('button_press_event', self.on_press),
            self.fig.canvas.mpl_connect('button_release_event', self.on_release),
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        ]

    def disconnect(self):
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)

    def calculate_continuum_for_centering(self, frame, y_min, y_max):
        full_image_size_X = frame.shape[1]
        all_x_indices = np.arange(full_image_size_X)
        y_indices = np.arange(y_min, y_max)

        sub_frame = frame[y_min:y_max, :]
        flux = np.sum(sub_frame, axis=0)

        valid = flux > 0
        centroids = np.zeros(full_image_size_X)
        if not np.any(valid) or y_min >= y_max:
            return np.full(full_image_size_X, (y_min + y_max) / 2.0)

        centroids[valid] = np.sum(sub_frame[:, valid] * y_indices[:, None], axis=0) / flux[valid]

        if np.sum(valid) < 10:
            return np.full(full_image_size_X, np.mean(centroids[valid]))

        p1 = np.polyfit(all_x_indices[valid], centroids[valid], 2)
        fit1 = np.polyval(p1, all_x_indices)
        residuals = centroids[valid] - fit1[valid]
        std_res = np.std(residuals)
        mask_clean = np.abs(centroids - fit1) <= 3.0 * std_res
        mask_final = valid & mask_clean
        if np.sum(mask_final) < 10: mask_final = valid

        clean_centroids_interp = np.interp(all_x_indices, all_x_indices[mask_final], centroids[mask_final])
        fit_final = rolling_mean(clean_centroids_interp, self.app.window_size)
        return fit_final

    def on_press(self, event):
        # Processing the right mouse button
        if event.button == 3:
            if event.inaxes in [self.ax_spec, self.ax_dev, self.ax_hd]:
                for line in self.sync_vlines:
                    line.remove()
                self.sync_vlines = [ax.axvline(event.xdata, color='green', linestyle='--', alpha=0.7)
                                    for ax in [self.ax_spec, self.ax_dev, self.ax_hd]]
                self.fig.canvas.draw_idle()
                return

        if event.inaxes != self.ax_img: return
        tol_y = self.master_1.shape[0] * 0.03
        tol_x = self.master_1.shape[1] * 0.03

        for line in self.h_lines:
            if abs(event.ydata - line.get_ydata()[0]) < tol_y:
                self.dragging_line = line
                self.kind = 'h'
                return
        for line in self.v_lines:
            if abs(event.xdata - line.get_xdata()[0]) < tol_x:
                self.dragging_line = line
                self.kind = 'v'
                return

    def on_motion(self, event):
        if self.dragging_line is None or event.inaxes != self.ax_img: return

        if self.kind == 'v':
            self.dragging_line.set_xdata([event.xdata, event.xdata])

        y1_current, y2_current = self.h_lines[0].get_ydata()[0], self.h_lines[1].get_ydata()[0]

        if self.kind == 'h':
            other_line = self.h_lines[0] if self.dragging_line is self.h_lines[1] else self.h_lines[1]
            y_other = other_line.get_ydata()[0]
            y_dragged = event.ydata
            width = abs(y_dragged - y_other)
            temp_y_min, temp_y_max = int(min(y_dragged, y_other)), int(max(y_dragged, y_other))
        else:
            width = abs(y1_current - y2_current)
            temp_y_min, temp_y_max = int(min(y1_current, y2_current)), int(max(y1_current, y2_current))

        if temp_y_min >= temp_y_max: temp_y_max = temp_y_min + 1

        if self.v_lines:
            x1, x2 = self.v_lines[0].get_xdata()[0], self.v_lines[1].get_xdata()[0]
            x_center_pix = int((x1 + x2) / 2)
        else:
            x_center_pix = self.master_1.shape[1] // 2

        continuum = self.calculate_continuum_for_centering(self.master_1, temp_y_min, temp_y_max)
        x_center_pix = np.clip(x_center_pix, 0, len(continuum) - 1)
        y_center_target = continuum[x_center_pix]

        new_y1, new_y2 = y_center_target - width / 2.0, y_center_target + width / 2.0
        self.h_lines[0].set_ydata([new_y1, new_y1])
        self.h_lines[1].set_ydata([new_y2, new_y2])

        self.update_plots()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.dragging_line = None

    def update_plots(self):
        y1, y2 = int(self.h_lines[0].get_ydata()[0]), int(self.h_lines[1].get_ydata()[0])
        self.app.current_ymin = max(0, min(y1, y2))
        self.app.current_ymax = min(self.master_1.shape[0], max(y1, y2) + 1)
        if self.app.current_ymin == self.app.current_ymax: self.app.current_ymax = self.app.current_ymin + 1

        if self.v_lines:
            x1, x2 = int(self.v_lines[0].get_xdata()[0]), int(self.v_lines[1].get_xdata()[0])
            xmin, xmax = max(0, min(x1, x2)), min(self.master_1.shape[1] - 1, max(x1, x2))
            if xmin == xmax: xmax = xmin + 1
        else:
            xmin, xmax = 0, self.master_1.shape[1] - 1

        plot_x = self.app.wavelengths[xmin:xmax + 1]

        new_spec1 = np.sum(self.master_1[self.app.current_ymin:self.app.current_ymax, xmin:xmax + 1], axis=0)
        new_spec2 = np.sum(self.master_2[self.app.current_ymin:self.app.current_ymax, xmin:xmax + 1], axis=0)

        if len(new_spec1) > 0 and len(new_spec2) > 0:
            max1 = np.max(new_spec1)
            max2 = np.max(new_spec2)

            if max1 != 0 and max2 != 0:
                amp = (max1 + max2) / 2.0

                new_spec1 = (new_spec1 / max1) * amp
                new_spec2 = (new_spec2 / max2) * amp

        self.spec_line1.set_data(plot_x, new_spec1)
        self.spec_line2.set_data(plot_x, new_spec2)

        dev1_full, err1_full, _ = self.app.calc_centroid_deviations_master(
            self.master_1, self.app.current_ymin, self.app.current_ymax, self.app.n_frames_1)
        dev2_full, err2_full, _ = self.app.calc_centroid_deviations_master(
            self.master_2, self.app.current_ymin, self.app.current_ymax, self.app.n_frames_2)

        dev1 = dev1_full[xmin:xmax + 1]
        err1 = err1_full[xmin:xmax + 1]
        dev2 = dev2_full[xmin:xmax + 1]
        err2 = err2_full[xmin:xmax + 1]

        med_sm = self.app.med_sm
        if med_sm > 1:
            plot_x_dev = np.array([np.median(plot_x[i:i + med_sm]) for i in range(0, len(plot_x), med_sm)])
            dev1 = np.array([np.median(dev1[i:i + med_sm]) for i in range(0, len(dev1), med_sm)])
            dev2 = np.array([np.median(dev2[i:i + med_sm]) for i in range(0, len(dev2), med_sm)])
            # Error smoothing (approximation via the median of errors / sqrt(med_sm))
            err1 = np.array([np.median(err1[i:i + med_sm]) for i in range(0, len(err1), med_sm)]) / np.sqrt(med_sm)
            err2 = np.array([np.median(err2[i:i + med_sm]) for i in range(0, len(err2), med_sm)]) / np.sqrt(med_sm)
        else:
            plot_x_dev = plot_x

        self.dev_line1.set_data(plot_x_dev, dev1)
        self.dev_line2.set_data(plot_x_dev, dev2)

        half_diff = (dev1 - dev2) / 2.0
        hd_err = np.sqrt(err1 ** 2 + err2 ** 2) / 2.0
        self.hd_line.set_data(plot_x_dev, half_diff)

        # Cleaning the old error interval fill
        if self.dev_fill1 is not None:
            self.dev_fill1.remove()
            self.dev_fill1 = None
        if self.dev_fill2 is not None:
            self.dev_fill2.remove()
            self.dev_fill2 = None
        if self.hd_fill is not None:
            self.hd_fill.remove()
            self.hd_fill = None

        # Rendering a new fill if the option is enabled
        if self.app.var_see_err.get() == 'YES':
            self.dev_fill1 = self.ax_dev.fill_between(plot_x_dev, dev1 - err1, dev1 + err1, color='C0', alpha=0.3)
            self.dev_fill2 = self.ax_dev.fill_between(plot_x_dev, dev2 - err2, dev2 + err2, color='C1', alpha=0.3)
            self.hd_fill = self.ax_hd.fill_between(plot_x_dev, half_diff - hd_err, half_diff + hd_err, color='gray',
                                                   alpha=0.3)

        # The current value of sigma from the menu
        try:
            current_sigma = float(self.app.param_vars['ax_y_sigma'].get())
        except:
            current_sigma = 3.0

        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_dev.relim()
        self.ax_dev.autoscale_view(scaley=False)
        std_dev = np.std(np.concatenate([dev1, dev2])) if len(dev1) > 0 else 0
        if std_dev > 0: self.ax_dev.set_ylim(-current_sigma * std_dev, current_sigma * std_dev)

        self.ax_hd.relim()
        self.ax_hd.autoscale_view(scaley=False)
        std_hd = np.std(half_diff) if len(half_diff) > 0 else 0
        if std_hd > 0: self.ax_hd.set_ylim(-current_sigma * std_hd, current_sigma * std_hd)


# The main class of the application
class SpectroAstrometryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPARTA (Spectral Processing for Analysis, Reduction and Two-source Astrometry)")

        # Adjusting the window size to fit the screen size
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.85)
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        self.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

        # Interception of the window closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.child_processes = []

        self.pa1_vars = {}  # Dictionaries for storing checkbox states
        self.pa2_vars = {}

        self.master_1 = None
        self.master_2 = None
        self.interactive = None

        self.n_frames_1 = 1
        self.n_frames_2 = 1

        self.create_widgets()
        self.setup_plot()

    def on_closing(self):
        for p in self.child_processes:
            if p.poll() is None:
                p.terminate()
        self.destroy()
        sys.exit(0)

    def create_scrollable_list(self, parent):
        container = tk.Frame(parent)
        canvas = tk.Canvas(container, height=100, width=350)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        container.pack(fill=tk.X, pady=2)

        return scrollable_frame, canvas

    def update_plot_if_interactive(self):
        if self.interactive:
            self.interactive.update_plots()
            self.canvas.draw_idle()

    def create_widgets(self):
        # Left control panel
        container = tk.Frame(self)
        container.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.control_frame = tk.Frame(canvas)

        self.control_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        control_frame = self.control_frame

        #1. Uploading files
        file_frame = tk.LabelFrame(control_frame, text="1. Load Files", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)

        tk.Button(file_frame, text="Load PA1", command=lambda: self.load_files(1)).pack(fill=tk.X, pady=2)
        self.frame_pa1, self.canvas_pa1 = self.create_scrollable_list(file_frame)

        tk.Button(file_frame, text="Load PA2", command=lambda: self.load_files(2)).pack(fill=tk.X, pady=2)
        self.frame_pa2, self.canvas_pa2 = self.create_scrollable_list(file_frame)

        # 2. FITS parameters
        fits_frame = tk.LabelFrame(control_frame, text="2. FITS Header Overrides", padx=5, pady=5)
        fits_frame.pack(fill=tk.X, pady=5)

        self.fits_vars = {}
        for text, default in [("NAXIS1", "FITS"), ("NAXIS2", "FITS"), ("CRVAL1", "FITS"),
                              ("CDELT1", "FITS")]:
            row = tk.Frame(fits_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=text, width=12, anchor='w').pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            tk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)
            self.fits_vars[text] = var

        # 3. Basic parameters
        params_frame = tk.LabelFrame(control_frame, text="3. Processing Parameters", padx=5, pady=5)
        params_frame.pack(fill=tk.X, pady=5)

        self.param_vars = {}
        defaults = {"Y_est": 218, "area": 20, "wl_start": 6500, "wl_end": 6900, "vmin": 2, "vmax": 98,
                    "window_size": 200, "med_sm": 1, "ax_y_sigma": 10, "proc_left": 0, "proc_right": 0}
        for text, val in defaults.items():
            row = tk.Frame(params_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=text, width=12, anchor='w').pack(side=tk.LEFT)
            var = tk.StringVar(value=str(val))
            tk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)
            self.param_vars[text] = var

        # 4. Options
        opts_frame = tk.LabelFrame(control_frame, text="4. Options", padx=5, pady=5)
        opts_frame.pack(fill=tk.X, pady=5)

        row1 = tk.Frame(opts_frame)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="rm_bg:", width=12, anchor='w').pack(side=tk.LEFT)
        self.var_rm_bg = ttk.Combobox(row1, values=['yes', 'no'], width=7, state="readonly")
        self.var_rm_bg.set('yes')
        self.var_rm_bg.pack(side=tk.RIGHT)

        row2 = tk.Frame(opts_frame)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="wl_borders:", width=12, anchor='w').pack(side=tk.LEFT)
        self.var_wl_borders = ttk.Combobox(row2, values=['True', 'False'], width=7, state="readonly")
        self.var_wl_borders.set('True')
        self.var_wl_borders.pack(side=tk.RIGHT)

        row3 = tk.Frame(opts_frame)
        row3.pack(fill=tk.X, pady=2)
        tk.Label(row3, text="see_err:", width=12, anchor='w').pack(side=tk.LEFT)
        self.var_see_err = ttk.Combobox(row3, values=['YES', 'NO'], width=7, state="readonly")
        self.var_see_err.set('YES')
        self.var_see_err.pack(side=tk.RIGHT)
        self.var_see_err.bind("<<ComboboxSelected>>", lambda e: self.update_plot_if_interactive())

        # 5. Control buttons
        btn_frame = tk.Frame(control_frame, pady=10)
        btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(btn_frame, text="Start", command=self.start_processing, bg="gray80", height=2).pack(fill=tk.X,
                                                                                                      pady=2)
        tk.Button(btn_frame, text="Reset", command=self.reset_plot, bg="gray80", height=2).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Save Data", command=self.save_data, bg="gray80").pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Analyze Data", command=self.analyze_data, bg="gray80").pack(fill=tk.X, pady=2)

        # Right panel for graphs and images
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def setup_plot(self):
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_files(self, pa_num):
        files = filedialog.askopenfilenames(title=f"Select FITS files for PA{pa_num}",
                                            filetypes=[("FITS files", "*.fts *.fits")])
        if not files: return

        target_frame = self.frame_pa1 if pa_num == 1 else self.frame_pa2
        target_vars = self.pa1_vars if pa_num == 1 else self.pa2_vars
        canvas = self.canvas_pa1 if pa_num == 1 else self.canvas_pa2

        # Cleaning up old data from the UI and dictionary
        for widget in target_frame.winfo_children():
            widget.destroy()
        target_vars.clear()

        # Checkmarks for each selected file
        for f in files:
            var = tk.BooleanVar(value=True)  # The checkmark is by default
            target_vars[f] = var
            filename = os.path.basename(f)
            chk = tk.Checkbutton(target_frame, text=filename, variable=var)
            chk.pack(anchor='w')

        # Updating the scroll area
        canvas.yview_moveto(0)

    def get_val(self, val_dict, key, typ=int):
        return typ(val_dict[key].get())

    def get_fits_val(self, key, header, default, typ=int):
        val = self.fits_vars[key].get().strip().upper()
        if val == 'FITS':
            return typ(header.get(key, default))
        return typ(val)

    def reset_plot(self):
        self.fig.clf()
        self.canvas.draw()
        if self.interactive:
            self.interactive.disconnect()
            self.interactive = None
        self.master_1 = None
        self.master_2 = None

    def load_raw_cube(self, file_paths, bg_mask, rm_bg):
        cube = []
        for path in file_paths:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(float)
                if rm_bg == 'yes':
                    bg_median = np.median(data[bg_mask, :], axis=0)
                    data -= bg_median
                cube.append(data)
        return np.array(cube)

    def get_y_centroid(self, frame, y_min, y_max):
        sub_frame = frame[y_min:y_max, :]
        flux = np.sum(sub_frame, axis=1)
        y_indices = np.arange(y_min, y_max)
        valid = flux > 0
        if np.sum(valid) == 0: return (y_min + y_max) / 2.0
        return np.sum(flux[valid] * y_indices[valid]) / np.sum(flux[valid])

    def calc_centroid_deviations_master(self, master_frame, y_min, y_max, n_frames=1):
        full_image_size_X = master_frame.shape[1]
        all_x_indices = np.arange(full_image_size_X)
        y_indices = np.arange(y_min, y_max)

        sub_frame = master_frame[y_min:y_max, :]
        flux = np.sum(sub_frame, axis=0)

        valid = flux > 0
        centroids = np.zeros(full_image_size_X)
        errors = np.zeros(full_image_size_X)

        if np.any(valid):
            centroids[valid] = np.sum(sub_frame[:, valid] * y_indices[:, None], axis=0) / flux[valid]

            # Error calculation: 0.6 * FWHM / SNR / sqrt(N_frames)
            variance = np.sum(sub_frame[:, valid] * (y_indices[:, None] - centroids[valid]) ** 2, axis=0) / flux[valid]
            sigma_y = np.sqrt(np.maximum(variance, 0))
            fwhm = 2.355 * sigma_y
            snr = np.sqrt(np.maximum(flux[valid], 1))
            errors[valid] = (0.6 * fwhm / snr) / np.sqrt(n_frames)

        p1 = np.polyfit(all_x_indices[valid], centroids[valid], 2)
        fit1 = np.polyval(p1, all_x_indices)
        residuals = centroids[valid] - fit1[valid]
        std_res = np.std(residuals)

        mask_clean = np.abs(centroids - fit1) <= 1.0 * std_res
        mask_final = valid & mask_clean

        clean_centroids_interp = np.interp(all_x_indices, all_x_indices[mask_final], centroids[mask_final])
        fit_final = rolling_mean(clean_centroids_interp, self.window_size)

        dev = np.zeros(full_image_size_X)
        dev[valid] = centroids[valid] - fit_final[valid]
        return dev, errors, fit_final

    def start_processing(self):
        # Lists only from files with checkmarks
        active_paths_1 = [f for f, var in self.pa1_vars.items() if var.get()]
        active_paths_2 = [f for f, var in self.pa2_vars.items() if var.get()]

        if not active_paths_1 or not active_paths_2:
            messagebox.showerror("Error", "Please select at least one file for both PA1 and PA2.")
            return

        self.n_frames_1 = len(active_paths_1)
        self.n_frames_2 = len(active_paths_2)

        self.reset_plot()

        try:
            # Reading parameters
            Y_est = self.get_val(self.param_vars, 'Y_est')
            area = self.get_val(self.param_vars, 'area')
            wl_start = self.get_val(self.param_vars, 'wl_start', float)
            wl_end = self.get_val(self.param_vars, 'wl_end', float)
            vmin = self.get_val(self.param_vars, 'vmin')
            vmax = self.get_val(self.param_vars, 'vmax')
            self.window_size = self.get_val(self.param_vars, 'window_size')
            self.med_sm = self.get_val(self.param_vars, 'med_sm')
            self.ax_y_sigma = self.get_val(self.param_vars, 'ax_y_sigma', float)
            proc_left = self.get_val(self.param_vars, 'proc_left')
            proc_right = self.get_val(self.param_vars, 'proc_right')
            rm_bg = self.var_rm_bg.get()
            wl_borders = self.var_wl_borders.get()

            # Reading the header from the first active PA1 file
            with fits.open(active_paths_1[0]) as hdul:
                header = hdul[0].header

            NAXIS1 = self.get_fits_val('NAXIS1', header, header.get('NAXIS1', 2048))
            NAXIS2 = self.get_fits_val('NAXIS2', header, header.get('NAXIS2', 512))
            CRVAL1 = self.get_fits_val('CRVAL1', header, 0, float)
            CDELT1 = self.get_fits_val('CDELT1', header, 1, float)

            self.wavelengths = CRVAL1 + np.arange(NAXIS1) * CDELT1

            bg_mask = np.ones(NAXIS2, dtype=bool)
            y_min_init = max(0, Y_est - area)
            y_max_init = min(NAXIS2, Y_est + area + 1)
            bg_mask[y_min_init: y_max_init] = False

            raw_cube_1 = self.load_raw_cube(active_paths_1, bg_mask, rm_bg)
            raw_cube_2 = self.load_raw_cube(active_paths_2, bg_mask, rm_bg)

            # Calculation of processing boundaries by X
            if proc_left + proc_right >= NAXIS1:
                x_min_proc, x_max_proc = 0, NAXIS1
            else:
                x_min_proc = proc_left
                x_max_proc = NAXIS1 - proc_right

            ref_frame = raw_cube_1[0]
            # Selecting an area for reference frame calculations
            ref_frame_sliced = ref_frame[:, x_min_proc:x_max_proc]
            ref_y_cent = self.get_y_centroid(ref_frame_sliced, y_min_init, y_max_init)
            ref_spec_x = np.sum(ref_frame_sliced[y_min_init:y_max_init, :], axis=0)

            def align_frame(frame):
                frame_sliced = frame[:, x_min_proc:x_max_proc]
                cent_y = self.get_y_centroid(frame_sliced, y_min_init, y_max_init)
                dy = ref_y_cent - cent_y

                # Y-shift the entire frame to avoid losing information at the edges
                frame_y_shifted = nd_shift(frame, (dy, 0), order=1, mode='nearest')

                # The spectrum is taken only within the specified X interval to search for an X shift
                spec_x = np.sum(frame_y_shifted[y_min_init:y_max_init, x_min_proc:x_max_proc], axis=0)

                def err_func(dx):
                    shifted_spec = nd_shift(spec_x, dx[0], order=1, mode='nearest')
                    return np.sum((shifted_spec - ref_spec_x) ** 2)

                res = minimize(err_func, x0=[0.0], method='Nelder-Mead')
                dx = res.x[0]

                # The final shift is applied to the entire frame
                return nd_shift(frame, (dy, dx), order=3, mode='nearest')

            self.master_1 = np.mean([align_frame(f) for f in raw_cube_1], axis=0)
            self.master_2 = np.mean([align_frame(f) for f in raw_cube_2], axis=0)

            # Visualization
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2])
            self.ax_img = self.fig.add_subplot(gs[0, 0])
            self.ax_spec = self.fig.add_subplot(gs[0, 1])
            self.ax_hd = self.fig.add_subplot(gs[1, 0])
            self.ax_dev = self.fig.add_subplot(gs[1, 1], sharex=self.ax_spec)

            for ax in [self.ax_img, self.ax_spec, self.ax_hd, self.ax_dev]:
                ax.tick_params(which='major', direction='in', length=0)

            self.ax_img.imshow(self.master_1, origin='lower', aspect='auto', cmap='gray',
                               vmin=np.percentile(self.master_1, vmin), vmax=np.percentile(self.master_1, vmax))

            self.h_lines = [
                self.ax_img.axhline(y_min_init, color='red', linestyle='dashed', linewidth=1.5),
                self.ax_img.axhline(y_max_init - 1, color='red', linestyle='dashed', linewidth=1.5)
            ]

            self.v_lines = []
            if wl_borders == 'True':
                valid_idx = (self.wavelengths >= min(wl_start, wl_end)) & (self.wavelengths <= max(wl_start, wl_end))
                x_indices = np.where(valid_idx)[0]
                if len(x_indices) > 0:
                    x_start, x_end = x_indices[0], x_indices[-1]
                    self.v_lines.extend([
                        self.ax_img.axvline(x_start, color='cyan', linestyle='dashed', linewidth=1.5),
                        self.ax_img.axvline(x_end, color='cyan', linestyle='dashed', linewidth=1.5)
                    ])

            # Drawing non-movable processing boundaries if the padding is not zero
            if proc_left > 0:
                self.ax_img.axvline(x_min_proc, color='purple', linestyle='-', linewidth=1.5)
            if proc_right > 0:
                self.ax_img.axvline(x_max_proc, color='purple', linestyle='-', linewidth=1.5)

            self.ax_img.set_title('Frame preview')
            self.ax_img.set_xlabel('X, pix')
            self.ax_img.set_ylabel('Y, pix')

            self.spec_line1, = self.ax_spec.plot([], [], color='C0', linewidth=1.5, label='PA1')
            self.spec_line2, = self.ax_spec.plot([], [], color='C1', linewidth=1.5, label='PA2')
            self.ax_spec.set_title('Averaged spectra')
            self.ax_spec.set_xlabel('Wavelength')
            self.ax_spec.set_ylabel(r'$<F_{comm}>$')
            self.ax_spec.legend()
            self.ax_spec.grid(True)

            self.dev_line1, = self.ax_dev.plot([], [], color='C0', label='PA1', linewidth=1.2)
            self.dev_line2, = self.ax_dev.plot([], [], color='C1', label='PA2 (+180°)', linewidth=1.5)
            self.ax_dev.axhline(0, color='black', linestyle='--', linewidth=0.8)
            self.ax_dev.set_title('Centroid deviations')
            self.ax_dev.set_xlabel('Wavelength')
            self.ax_dev.set_ylabel('Offset, pix')
            self.ax_dev.grid(True)

            self.hd_line, = self.ax_hd.plot([], [], color='gray', linewidth=1.5)
            self.ax_hd.axhline(0, color='black', linestyle='--', linewidth=0.8)
            self.ax_hd.set_title('Spectroastrometric signal')
            self.ax_hd.set_xlabel('Wavelength')
            self.ax_hd.set_ylabel('Offset, pix')
            self.ax_hd.grid(True)

            self.fig.tight_layout()

            self.interactive = InteractiveSpectroAstrometry(self)
            self.interactive.update_plots()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")

    def save_data(self):
        if self.master_1 is None or self.master_2 is None:
            messagebox.showwarning("Warning", "No processed data available. Press Start first.")
            return False

        try:
            os.makedirs('data', exist_ok=True)
            fits.PrimaryHDU(self.master_1).writeto('data/master_1.fits', overwrite=True)
            fits.PrimaryHDU(self.master_2).writeto('data/master_2.fits', overwrite=True)

            full_spec_1 = np.sum(self.master_1[self.current_ymin:self.current_ymax, :], axis=0)
            full_spec_2 = np.sum(self.master_2[self.current_ymin:self.current_ymax, :], axis=0)

            full_dev_1, full_err_1, traj_1 = self.calc_centroid_deviations_master(self.master_1, self.current_ymin,
                                                                                  self.current_ymax, self.n_frames_1)
            full_dev_2, full_err_2, traj_2 = self.calc_centroid_deviations_master(self.master_2, self.current_ymin,
                                                                                  self.current_ymax, self.n_frames_2)

            full_hd = (full_dev_1 - full_dev_2) / 2.0
            full_hd_err = np.sqrt(full_err_1 ** 2 + full_err_2 ** 2) / 2.0

            # Data for the table
            data_dict = {
                'Wavelength': self.wavelengths,
                'Flux_PA1': full_spec_1,
                'Flux_PA2': full_spec_2,
                'Dev_PA1': full_dev_1,
                'Err_PA1': full_err_1,
                'Dev_PA2': full_dev_2,
                'Err_PA2': full_err_2,
                'Half_Diff': full_hd,
                'Err_Half_Diff': full_hd_err,
                'Cont_PA1': traj_1,
                'Cont_PA2': traj_2
            }

            df = pd.DataFrame(data_dict)

            params = {k: v.get() for k, v in self.param_vars.items()}
            params.update({k: v.get() for k, v in self.fits_vars.items()})
            params['rm_bg'] = self.var_rm_bg.get()
            params['wl_borders'] = self.var_wl_borders.get()
            params['see_err'] = self.var_see_err.get()
            params['final_y_min'] = self.current_ymin
            params['final_y_max'] = self.current_ymax

            # Write selected files in comments
            params['Active_PA1_Files'] = ", ".join([os.path.basename(f) for f, v in self.pa1_vars.items() if v.get()])
            params['Active_PA2_Files'] = ", ".join([os.path.basename(f) for f, v in self.pa2_vars.items() if v.get()])

            csv_path = 'data/params_and_data.csv'
            with open(csv_path, 'w') as f:
                for k, v in params.items():
                    f.write(f"# {k}: {v}\n")
            df.to_csv(csv_path, mode='a', index=False)

            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data:\n{str(e)}")
            return False

    def analyze_data(self):
        # If the data is present in the current session, it is saved before analysis
        if self.master_1 is not None and self.master_2 is not None:
            if not self.save_data():
                return
        else:
            # If the session is empty, the existence of previously saved files is checked
            required_files = ['data/master_1.fits', 'data/master_2.fits', 'data/params_and_data.csv']
            if not all(os.path.exists(f) for f in required_files):
                messagebox.showwarning("Warning", "No processed data found in 'data' folder. Press Start first.")
                return

        # Launching the next stage
        try:
            if getattr(sys, 'frozen', False):
                p = subprocess.Popen([sys.executable, "processing"])
                self.child_processes.append(p)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                target_script = os.path.join(script_dir, 'SPARTA_processing.py')

                if not os.path.exists(target_script):
                    messagebox.showerror("Error", f"File not found: {target_script}")
                    return

                p = subprocess.Popen([sys.executable, target_script])
                self.child_processes.append(p)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch processing:\n{str(e)}")


if __name__ == "__main__":
    app = SpectroAstrometryApp()
    app.mainloop()
