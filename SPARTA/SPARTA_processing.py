import os
import sys
import csv
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit


def gaussian(x, a, mu, sigma, offset):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset


def fit_profile_center(y_coords, profile):
    a_guess = np.max(profile) - np.min(profile)
    mu_guess = y_coords[np.argmax(profile)]
    sigma_guess = 2.0
    offset_guess = np.min(profile)

    try:
        popt, _ = curve_fit(gaussian, y_coords, profile,
                            p0=[a_guess, mu_guess, sigma_guess, offset_guess],
                            maxfev=1000)
        return popt[1]  # Returns mu
    except:
        return np.nan  # If the approximation fails


# The main class of the application
class SAHeatmapsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPARTA (Spectral Processing for Analysis, Reduction and Two-source Astrometry)")

        # Adjusting window sizes
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w, win_h = int(screen_w * 0.8), int(screen_h * 0.7)
        self.root.geometry(f"{win_w}x{win_h}")

        # Storage for raw and filtered maps
        self.raw_map1 = None
        self.raw_map2 = None
        self.map1_filtered = None
        self.map2_filtered = None
        self.map3_filtered = None
        self.map_extent = None

        # Offset table data storage
        self.table_rows = []
        self.final_d1 = np.nan
        self.final_d1_err = np.nan
        self.final_d2 = np.nan
        self.final_d2_err = np.nan
        self.final_sep = np.nan
        self.final_sep_err = np.nan

        self.load_data()

        self.setup_ui()
        self.update_top_plot()

    def load_data(self):
        csv_path = 'data/params_and_data.csv'
        fits1_path = 'data/master_1.fits'
        fits2_path = 'data/master_2.fits'

        if not all(os.path.exists(p) for p in [csv_path, fits1_path, fits2_path]):
            messagebox.showerror("Error", "Files not found in 'data/' directory")
            sys.exit()

        # Parameter parsing
        self.params = {}
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    parts = line[1:].strip().split(": ", 1)
                    if len(parts) == 2: self.params[parts[0]] = parts[1]

        self.df = pd.read_csv(csv_path, comment='#')

        # Loading 2D data
        self.data_pa1 = fits.getdata(fits1_path)
        self.data_pa2 = fits.getdata(fits2_path)

        # Extracting data arrays
        self.wl = self.df['Wavelength'].values
        self.flux1 = self.df['Flux_PA1'].values
        self.flux2 = self.df['Flux_PA2'].values
        self.hd = self.df['Half_Diff'].values
        self.hd_err = self.df['Err_Half_Diff'].values

        # Loading the continuum arrays
        self.cont_pa1 = self.df['Cont_PA1'].values
        self.cont_pa2 = self.df['Cont_PA2'].values

        # Adjusting the spectra to the range from the previous step
        self.wl_start = float(self.params.get('wl_start', self.wl.min()))
        self.wl_end = float(self.params.get('wl_end', self.wl.max()))
        self.wl_start, self.wl_end = sorted([self.wl_start, self.wl_end])

        self.y_est = float(self.params.get('Y_est', self.data_pa1.shape[0] // 2))
        self.default_area = int(self.params.get('area', 10))

    def setup_ui(self):
        # Divided into left (panel) and right (pictures) parts
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left panel with scrollbar
        self.left_frame = tk.Frame(self.main_pane, width=320)
        self.main_pane.add(self.left_frame, minsize=300)

        canvas = tk.Canvas(self.left_frame)
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.frame_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.frame_id, width=e.width))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        center_wl = (self.wl_start + self.wl_end) / 2
        self.var_line = tk.StringVar(value=f"{center_wl:.2f}")
        self.var_dcont = tk.StringVar(value="10.0")
        self.var_prof_area = tk.StringVar(value=str(self.default_area))
        self.var_indent = tk.StringVar(value="1")

        self.var_min_pa1 = tk.StringVar(value="1")
        self.var_max_pa1 = tk.StringVar(value="99")
        self.var_min_pa2 = tk.StringVar(value="1")
        self.var_max_pa2 = tk.StringVar(value="99")

        self.var_d1_val = tk.StringVar()
        self.var_d1_err = tk.StringVar(value="0")
        self.var_d2_val = tk.StringVar()
        self.var_d2_err = tk.StringVar(value="0")
        self.var_sep_val = tk.StringVar()
        self.var_sep_err = tk.StringVar(value="0")

        self.var_line.trace_add("write", lambda *args: self.update_top_plot())
        self.var_dcont.trace_add("write", lambda *args: self.update_top_plot())

        self.var_min_pa1.trace_add("write", lambda *args: self.apply_thresholds_and_draw())
        self.var_max_pa1.trace_add("write", lambda *args: self.apply_thresholds_and_draw())
        self.var_min_pa2.trace_add("write", lambda *args: self.apply_thresholds_and_draw())
        self.var_max_pa2.trace_add("write", lambda *args: self.apply_thresholds_and_draw())

        # 1. Processing Frames
        self.frame_proc = tk.LabelFrame(self.scrollable_frame, text="1. Processing Frames", font=("Arial", 10))
        self.frame_proc.pack(fill="x", padx=5, pady=5)

        fields = [
            ("line_center", self.var_line),
            ("d_cont", self.var_dcont),
            ("prof_area", self.var_prof_area),
            ("indent", self.var_indent),
            ("vmin_1", self.var_min_pa1),
            ("vmax_1", self.var_max_pa1),
            ("vmin_2", self.var_min_pa2),
            ("vmax_2", self.var_max_pa2)
        ]

        self.frame_proc.columnconfigure(0, weight=1)
        self.frame_proc.columnconfigure(1, weight=1)

        for i, (text, var) in enumerate(fields):
            tk.Label(self.frame_proc, text=text).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            tk.Entry(self.frame_proc, textvariable=var, width=12).grid(row=i, column=1, sticky='e', padx=5, pady=5)

        btn_calc = tk.Button(self.frame_proc, text="Create Map", command=self.create_maps, bg="gray80")
        btn_calc.grid(row=len(fields), column=0, columnspan=2, pady=10, padx=5, sticky="ew")

        # 2. Definition of Separation
        self.frame_sep = tk.LabelFrame(self.scrollable_frame, text="Definition of Separation", font=("Arial", 10))
        self.frame_sep.pack(fill="x", padx=5, pady=5)

        btn_write_row = tk.Button(self.frame_sep, text="Write Row", command=self.add_table_row, bg="gray80")
        btn_write_row.pack(fill="x", padx=5, pady=5)

        # Table headers
        header_frame = tk.Frame(self.frame_sep)
        header_frame.pack(fill="x", padx=5, pady=2)
        tk.Label(header_frame, text="λ", width=8, anchor="center").pack(side="left")
        tk.Label(header_frame, text="d, pix", width=8, anchor="center").pack(side="left")
        tk.Label(header_frame, text="Δd, pix", width=8, anchor="center").pack(side="left")
        tk.Label(header_frame, text="source", width=6, anchor="center").pack(side="left")

        # Container for table rows
        self.table_inner_frame = tk.Frame(self.frame_sep)
        self.table_inner_frame.pack(fill="x", padx=5, pady=2)

        btn_find_sep = tk.Button(self.frame_sep, text="Find Separation", command=self.calculate_separation, bg="gray80")
        btn_find_sep.pack(fill="x", padx=5, pady=10)

        # Fields for manual input/display of results for d1, d2, sep
        d1_input_frame = tk.Frame(self.frame_sep)
        d1_input_frame.pack(pady=2)
        tk.Label(d1_input_frame, text="d1 =", fg="black").pack(side="left")
        self.ent_d1_val = tk.Entry(d1_input_frame, textvariable=self.var_d1_val, width=8)
        self.ent_d1_val.pack(side="left", padx=2)
        tk.Label(d1_input_frame, text="±", fg="black").pack(side="left")
        self.ent_d1_err = tk.Entry(d1_input_frame, textvariable=self.var_d1_err, width=8)
        self.ent_d1_err.pack(side="left", padx=2)
        tk.Label(d1_input_frame, text="pix", fg="black").pack(side="left")

        d2_input_frame = tk.Frame(self.frame_sep)
        d2_input_frame.pack(pady=2)
        tk.Label(d2_input_frame, text="d2 =", fg="black").pack(side="left")
        self.ent_d2_val = tk.Entry(d2_input_frame, textvariable=self.var_d2_val, width=8)
        self.ent_d2_val.pack(side="left", padx=2)
        tk.Label(d2_input_frame, text="±", fg="black").pack(side="left")
        self.ent_d2_err = tk.Entry(d2_input_frame, textvariable=self.var_d2_err, width=8)
        self.ent_d2_err.pack(side="left", padx=2)
        tk.Label(d2_input_frame, text="pix", fg="black").pack(side="left")

        sep_input_frame = tk.Frame(self.frame_sep)
        sep_input_frame.pack(pady=2)
        tk.Label(sep_input_frame, text="sep =", fg="black").pack(side="left")
        self.ent_sep_val = tk.Entry(sep_input_frame, textvariable=self.var_sep_val, width=8)
        self.ent_sep_val.pack(side="left", padx=2)
        tk.Label(sep_input_frame, text="±", fg="black").pack(side="left")
        self.ent_sep_err = tk.Entry(sep_input_frame, textvariable=self.var_sep_err, width=8)
        self.ent_sep_err.pack(side="left", padx=2)
        tk.Label(sep_input_frame, text="pix", fg="black").pack(side="left")

        # 3. Save and start buttons
        self.frame_bottom = tk.Frame(self.scrollable_frame)
        self.frame_bottom.pack(fill="x", padx=5, pady=15)

        btn_save = tk.Button(self.frame_bottom, text="Save Data", command=self.save_data, bg="gray80")
        btn_save.pack(fill="x", pady=5)

        btn_separate = tk.Button(self.frame_bottom, text="Separate Spectra", command=self.run_separate_spectra,
                                 bg="gray80")
        btn_separate.pack(fill="x", pady=5)

        # Right panel
        self.right_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.right_frame, stretch="always")

        self.fig = plt.Figure(figsize=(12, 10))
        gs = GridSpec(3, 3, height_ratios=[1, 1, 2], figure=self.fig)

        self.ax_spec = self.fig.add_subplot(gs[0, :])
        self.ax_hd = self.fig.add_subplot(gs[1, :], sharex=self.ax_spec)
        self.ax_map1 = self.fig.add_subplot(gs[2, 0])
        self.ax_map2 = self.fig.add_subplot(gs[2, 1])
        self.ax_map3 = self.fig.add_subplot(gs[2, 2])

        self.fig.tight_layout(pad=3.0)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_plot.mpl_connect('button_press_event', self.on_click)

        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.right_frame)
        self.toolbar.update()

    def get_safe_float(self, var, default=0.0):
        try:
            return float(var.get())
        except ValueError:
            return default

    def update_top_plot(self):
        self.ax_spec.clear()
        self.ax_hd.clear()

        mask = (self.wl >= self.wl_start) & (self.wl <= self.wl_end)
        wl_m = self.wl[mask]

        self.ax_spec.plot(wl_m, self.flux1[mask] / np.max(self.flux1[mask]), label='PA1', color='C0')
        self.ax_spec.plot(wl_m, self.flux2[mask] / np.max(self.flux2[mask]), label='PA2', color='C1')
        self.ax_spec.set_title("Normalized spectra")
        self.ax_spec.set_ylabel("Flux")
        self.ax_spec.set_xlabel("Wavelength")
        self.ax_spec.legend()

        self.ax_hd.plot(wl_m, self.hd[mask], color='black', lw=1.2)
        self.ax_hd.fill_between(wl_m, self.hd[mask] - self.hd_err[mask], self.hd[mask] + self.hd_err[mask],
                                color='gray', alpha=0.3)
        self.ax_hd.axhline(0, color='red', ls='-', lw=0.8)
        self.ax_hd.set_title("Spectroastrometric signal")
        self.ax_hd.set_ylabel("Offset, pix")
        self.ax_hd.set_xlabel("Wavelength")

        line = self.get_safe_float(self.var_line)
        d_cont = self.get_safe_float(self.var_dcont)

        for ax in [self.ax_spec, self.ax_hd]:
            if line > 0:
                ax.axvline(line, color='red', ls='-', lw=1.5, label='Line' if ax == self.ax_spec else "")
                if d_cont > 0:
                    ax.axvline(line - d_cont, color='red', ls='--', lw=1)
                    ax.axvline(line + d_cont, color='red', ls='--', lw=1)
            ax.grid(True, alpha=0.3)

        self.ax_spec.set_xlim(self.wl_start, self.wl_end)
        self.canvas_plot.draw_idle()

    def get_pixel_index(self, wavelength):
        return np.argmin(np.abs(self.wl - wavelength))

    def create_maps(self):
        line_wl = self.get_safe_float(self.var_line)
        d_cont_wl = self.get_safe_float(self.var_dcont)
        prof_area = int(self.get_safe_float(self.var_prof_area, self.default_area))
        indent = int(self.get_safe_float(self.var_indent, 1))

        center_px = self.get_pixel_index(line_wl)
        start_px = self.get_pixel_index(line_wl - d_cont_wl)
        end_px = self.get_pixel_index(line_wl + d_cont_wl)

        start_px, end_px = sorted([start_px, end_px])
        pixel_range = range(start_px, end_px + 1)
        n_pixels = len(pixel_range)

        map1 = np.full((n_pixels, n_pixels), np.nan)
        map2 = np.full((n_pixels, n_pixels), np.nan)

        y_min = max(0, int(self.y_est - prof_area))
        y_max = min(self.data_pa1.shape[0], int(self.y_est + prof_area + 1))
        y_coords = np.arange(y_min, y_max)

        for i, idx_L in enumerate(pixel_range):
            dist_L = abs(idx_L - center_px)
            prof_L1 = self.data_pa1[y_min:y_max, idx_L]
            prof_L2 = self.data_pa2[y_min:y_max, idx_L]

            cont_val1 = self.cont_pa1[idx_L]
            cont_val2 = self.cont_pa2[idx_L]

            for j, idx_C in enumerate(pixel_range):
                dist_C = abs(idx_C - center_px)

                # Map filtering: the distance from line to L is less than the distance to C
                if (dist_C - dist_L) >= indent:
                    prof_C1 = self.data_pa1[y_min:y_max, idx_C]
                    prof_C2 = self.data_pa2[y_min:y_max, idx_C]

                    diff1 = np.abs(prof_L1 - prof_C1)
                    diff2 = np.abs(prof_L2 - prof_C2)

                    cen1 = fit_profile_center(y_coords, diff1)
                    cen2 = fit_profile_center(y_coords, diff2)

                    offset_pa1 = cen1 - cont_val1
                    offset_pa2 = cen2 - cont_val2

                    if not np.isnan(cen1): map1[i, j] = offset_pa1
                    if not np.isnan(cen2): map2[i, j] = offset_pa2

        self.raw_map1 = map1
        self.raw_map2 = map2

        wl_start = self.wl[start_px]
        wl_end = self.wl[end_px]
        self.map_extent = [wl_start, wl_end, wl_start, wl_end]

        self.apply_thresholds_and_draw()

    def apply_thresholds_and_draw(self):
        if self.raw_map1 is None or self.raw_map2 is None:
            return

        min_p1 = np.clip(self.get_safe_float(self.var_min_pa1, 1), 0, 100)
        max_p1 = np.clip(self.get_safe_float(self.var_max_pa1, 99), 0, 100)
        min_p2 = np.clip(self.get_safe_float(self.var_min_pa2, 1), 0, 100)
        max_p2 = np.clip(self.get_safe_float(self.var_max_pa2, 99), 0, 100)

        map1_filtered = np.copy(self.raw_map1)
        map2_filtered = np.copy(self.raw_map2)

        if not np.all(np.isnan(map1_filtered)):
            val_min1 = np.nanpercentile(map1_filtered, min_p1)
            val_max1 = np.nanpercentile(map1_filtered, max_p1)
            map1_filtered[(map1_filtered < val_min1) | (map1_filtered > val_max1)] = np.nan

        if not np.all(np.isnan(map2_filtered)):
            val_min2 = np.nanpercentile(map2_filtered, min_p2)
            val_max2 = np.nanpercentile(map2_filtered, max_p2)
            map2_filtered[(map2_filtered < val_min2) | (map2_filtered > val_max2)] = np.nan

        map3_filtered = (map1_filtered - map2_filtered) / 2.0

        # Saving filtered maps
        self.map1_filtered = map1_filtered
        self.map2_filtered = map2_filtered
        self.map3_filtered = map3_filtered

        self.draw_heatmaps(map1_filtered, map2_filtered, map3_filtered)

    def draw_heatmaps(self, map1, map2, map3):
        self.ax_map1.clear()
        self.ax_map2.clear()
        self.ax_map3.clear()

        im1 = self.ax_map1.imshow(map1, origin='lower', extent=self.map_extent, cmap='jet', aspect='auto')
        im2 = self.ax_map2.imshow(map2, origin='lower', extent=self.map_extent, cmap='jet', aspect='auto')
        im3 = self.ax_map3.imshow(map3, origin='lower', extent=self.map_extent, cmap='jet', aspect='auto')

        self.ax_map1.set_title("PA1")
        self.ax_map2.set_title("PA2 ")
        self.ax_map3.set_title("(PA1 - PA2) / 2")

        for ax in [self.ax_map1, self.ax_map2, self.ax_map3]:
            ax.set_xlabel(r"$λ_C$")
            ax.set_ylabel(r"$λ_L$")
            ax.grid(False)

        self.canvas_plot.draw_idle()

    def on_click(self, event):
        if event.button == 3 and event.inaxes in [self.ax_spec, self.ax_hd]:
            if event.xdata is not None:
                self.var_line.set(f"{event.xdata:.2f}")

    # Offset table logic
    def add_table_row(self):
        if self.map3_filtered is None:
            messagebox.showwarning("Warning", "Create Map first!")
            return

        valid_pixels = self.map3_filtered[~np.isnan(self.map3_filtered)]
        N = len(valid_pixels)

        if N == 0:
            messagebox.showwarning("Warning", "Map contains only NaN pixels!")
            return

        lam = self.get_safe_float(self.var_line)
        d_mean = np.mean(valid_pixels)
        d_err = np.std(valid_pixels) / np.sqrt(N) if N > 1 else 0.0

        row_frame = tk.Frame(self.table_inner_frame)
        row_frame.pack(fill="x", pady=2)

        tk.Label(row_frame, text=f"{lam:.2f}", width=8, anchor="center").pack(side="left")
        tk.Label(row_frame, text=f"{d_mean:.4f}", width=8, anchor="center").pack(side="left")
        tk.Label(row_frame, text=f"{d_err:.4f}", width=8, anchor="center").pack(side="left")

        source_var = tk.StringVar(value="1")
        cb = ttk.Combobox(row_frame, textvariable=source_var, values=["1", "2"], width=4, state="readonly")
        cb.pack(side="left", padx=5)

        row_data = {
            'lam': lam,
            'd': d_mean,
            'err': d_err,
            'source_var': source_var,
            'frame': row_frame
        }

        def delete_row():
            row_frame.destroy()
            if row_data in self.table_rows:
                self.table_rows.remove(row_data)

        btn_del = tk.Button(row_frame, text="✖", command=delete_row, fg="gray30", width=2)
        btn_del.pack(side="left")

        self.table_rows.append(row_data)

    def calculate_separation(self):
        if not self.table_rows:
            messagebox.showwarning("Warning", "Table is empty!")
            return

        src1_d, src1_w = [], []
        src2_d, src2_w = [], []

        for row in self.table_rows:
            src = row['source_var'].get()
            d = row['d']
            err = max(row['err'], 1e-6)  # Protection against division by 0
            w = 1.0 / (err ** 2)

            if src == "1":
                src1_d.append(d)
                src1_w.append(w)
            elif src == "2":
                src2_d.append(d)
                src2_w.append(w)

        def weighted_stats(vals, weights):
            if not vals: return np.nan, np.nan
            v, w = np.array(vals), np.array(weights)
            w_sum = np.sum(w)
            mean = np.sum(v * w) / w_sum
            err = 1.0 / np.sqrt(w_sum)
            return mean, err

        mean1, err1 = weighted_stats(src1_d, src1_w)
        mean2, err2 = weighted_stats(src2_d, src2_w)

        if not np.isnan(mean1):
            self.final_d1 = mean1
            self.final_d1_err = err1
            self.var_d1_val.set(f"{self.final_d1:.4f}")
            self.var_d1_err.set(f"{self.final_d1_err:.4f}")
        else:
            self.var_d1_val.set("")
            self.var_d1_err.set("0")

        if not np.isnan(mean2):
            self.final_d2 = mean2
            self.final_d2_err = err2
            self.var_d2_val.set(f"{self.final_d2:.4f}")
            self.var_d2_err.set(f"{self.final_d2_err:.4f}")
        else:
            self.var_d2_val.set("")
            self.var_d2_err.set("0")

        if not np.isnan(mean1) and not np.isnan(mean2):
            self.final_sep = abs(mean1 - mean2)
            self.final_sep_err = np.sqrt(err1 ** 2 + err2 ** 2)
            self.var_sep_val.set(f"{self.final_sep:.4f}")
            self.var_sep_err.set(f"{self.final_sep_err:.4f}")
        else:
            self.final_sep = np.nan
            self.final_sep_err = np.nan
            self.var_sep_val.set("")
            self.var_sep_err.set("0")

    # Saving data and starting the next step
    def save_data(self):
        os.makedirs('data', exist_ok=True)
        filepath = 'data/processing_data.csv'

        # Checking the availability of d1 and d2
        if not self.var_d1_val.get().strip() or not self.var_d2_val.get().strip():
            messagebox.showerror("Error",
                                 "Please define the 'd1' and 'd2' parameters and their errors before proceeding.")
            return

        # Updating values from input fields before saving
        try:
            d1_val_str = self.var_d1_val.get()
            d1_err_str = self.var_d1_err.get()
            self.final_d1 = float(d1_val_str) if d1_val_str else np.nan
            self.final_d1_err = float(d1_err_str) if d1_err_str else 0.0

            d2_val_str = self.var_d2_val.get()
            d2_err_str = self.var_d2_err.get()
            self.final_d2 = float(d2_val_str) if d2_val_str else np.nan
            self.final_d2_err = float(d2_err_str) if d2_err_str else 0.0

            val_str = self.var_sep_val.get()
            err_str = self.var_sep_err.get()
            self.final_sep = float(val_str) if val_str else np.nan
            self.final_sep_err = float(err_str) if err_str else 0.0
        except ValueError:
            pass

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # 1. Interface Parameters
            writer.writerow(["# PARAMETERS"])
            writer.writerow(["line_center", self.var_line.get()])
            writer.writerow(["d_cont", self.var_dcont.get()])
            writer.writerow(["prof_area", self.var_prof_area.get()])
            writer.writerow(["indent", self.var_indent.get()])
            writer.writerow(["vmin_1", self.var_min_pa1.get()])
            writer.writerow(["vmax_1", self.var_max_pa1.get()])
            writer.writerow(["vmin_2", self.var_min_pa2.get()])
            writer.writerow(["vmax_2", self.var_max_pa2.get()])
            writer.writerow([])

            #2. Final data
            writer.writerow(["# SEPARATION RESULTS"])
            writer.writerow(["d1", self.final_d1])
            writer.writerow(["d1_Err", self.final_d1_err])
            writer.writerow(["d2", self.final_d2])
            writer.writerow(["d2_Err", self.final_d2_err])
            writer.writerow(["Sep", self.final_sep])
            writer.writerow(["Sep_Err", self.final_sep_err])
            writer.writerow([])

            # 3. Data from the table
            writer.writerow(["# TABLE"])
            writer.writerow(["lam_A", "d_pix", "err_pix", "source"])
            for row in self.table_rows:
                writer.writerow([row['lam'], row['d'], row['err'], row['source_var'].get()])
            writer.writerow([])

            # 4. Color maps
            def write_map(name, map_data):
                writer.writerow([f"# {name}"])
                if map_data is not None:
                    for row in map_data:
                        writer.writerow(row)
                writer.writerow([])

            write_map("MAP1", self.map1_filtered)
            write_map("MAP2", self.map2_filtered)
            write_map("MAP3", self.map3_filtered)

    def run_separate_spectra(self):
        # Automatic saving
        self.save_data()

        # Checking the availability of d1 and d2
        if not self.var_d1_val.get().strip() or not self.var_d2_val.get().strip():
            messagebox.showerror("Error",
                                 "Please define the 'd1' and 'd2' parameters and their errors before proceeding.")
            return

        try:
            if getattr(sys, 'frozen', False):
                subprocess.Popen([sys.executable, "separation"])
            else:
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "SPARTA_separation.py")

                if not os.path.exists(script_path):
                    messagebox.showerror("Error", f"File not found: {script_path}")
                    return

                subprocess.Popen([sys.executable, script_path])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run script:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SAHeatmapsApp(root)
    root.mainloop()
