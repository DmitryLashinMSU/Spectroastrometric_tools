import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import os
import json

from Spectroastrometry import *
from SAModel import *

import sys
import csv

# Increase the CSV cell size limit to upload large data arrays
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 16,
})


class TalosApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TALOS (Test for Astrometric Limits of Observed Spectra)")

        # Setting the window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.59)
        self.root.geometry(f"{window_width}x{window_height}")

        # Closing the program when the GUI window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---- Non-configurable parameters ----

        #   These parameters were originally set up to handle real data,      #
        #   but now they serve as a basis for modeling. Don't look for        #
        #   deep physical meaning in them, because there isn't any. You       #
        #   can set up any desired model without modifying these parameters.  #
        #   "If it works, don't touch it!"                                    #

        self.image_size_Y = 512
        self.CRVAL1 = 0.0
        self.CDELT1 = 1.0
        self.start_idx = 1150
        self.end_idx = 1300
        self.image_size_X_2 = self.end_idx - self.start_idx
        self.crop = 220
        self.image_size_Y_2 = self.image_size_Y - 2 * self.crop
        self.Y_est_2 = 252 - self.crop
        self.delt = 7

        # ---- Configurable parameters ----
        self.param_vars = {}
        self.param_types = {
            'sp_level': (20000, int),  # Constant spectrum intensity (ADU)
            'area': (10, int),  # Half-width of the area across the spectrum in which it is analyzed
            'pix_size': (0.37, float),  # Pixel size (arcsec)
            'seeing': (1.0, float),  # Scale of the atmospheric PSF (arcsec)
            'slit': (1.0, float),  # Slit size (arcsec)
            'RN': (3, int),  # Readout noise (e/ADU)
            'BN': (5, int),  # Background (ADU)
            'N': (15, int),  # Number of lines in the test spectrum
            'N_obs': (3, int),  # Number of frames for each parameter set
            'I_cont': (0.0, float),  # Test spectrum continuum intensity (in fractions of sp_level)
            'I_min': (1e-4, float),  # Test spectrum left line intensity (in fractions of sp_level)
            'I_max': (1e-1, float),  # Test spectrum right line intensity (in fractions of sp_level)
            'sep_min': (0.01, float),  # Minimum angular separation of sources
            'sep_max': (0.5, float),  # Maximum angular separation of sources
            'threshold': (1.25, float),  # Signal detection threshold (sigma)
            'N_it': (20, int)  # Number of parameter sets (y-axis size)
        }

        # Session state variables
        self.session_data = None
        self.select_mode = False
        self.points = []

        self.setup_ui()

    def setup_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left panel with scrolling
        left_frame_container = tk.Frame(main_pane, width=300)
        main_pane.add(left_frame_container, minsize=300)

        canvas = tk.Canvas(left_frame_container, width=300)
        scrollbar = ttk.Scrollbar(left_frame_container, orient=tk.VERTICAL, command=canvas.yview)

        self.left_frame = tk.Frame(canvas)
        self.left_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        self.frame_id = canvas.create_window((0, 0), window=self.left_frame, anchor="nw")

        self.left_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        def on_canvas_configure(event):
            canvas.itemconfig(self.frame_id, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Input fields
        params_frame = tk.Frame(self.left_frame)
        params_frame.pack(fill=tk.X, padx=10)

        row_idx = 0
        for key, (default_val, val_type) in self.param_types.items():
            tk.Label(params_frame, text=key).grid(row=row_idx, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=str(default_val))
            self.param_vars[key] = var
            entry = tk.Entry(params_frame, textvariable=var, width=12, justify=tk.CENTER)
            entry.grid(row=row_idx, column=1, sticky=tk.W, pady=2)
            row_idx += 1

            # Dynamic display of FWHM when N is entered
            if key == 'N':
                self.fwhm_label = tk.Label(params_frame, text="", fg="black")
                self.fwhm_label.grid(row=row_idx, column=1, sticky=tk.W, pady=0)
                row_idx += 1
                var.trace_add('write', self.update_fwhm)

        self.update_fwhm()  # Initializing the FWHM value

        # Button panel
        btn_frame = tk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=20)

        tk.Button(btn_frame, text="Start", command=self.start_calculation).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="Save", command=self.save_data).pack(fill=tk.X, pady=2)

        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.btn_limit = tk.Button(btn_frame, text="Find Limit", command=self.find_limit, state=tk.DISABLED)
        self.btn_limit.pack(fill=tk.X, pady=2)

        self.btn_select = tk.Button(btn_frame, text="Select Data (OFF)", command=self.toggle_select, state=tk.DISABLED)
        self.btn_select.pack(fill=tk.X, pady=2)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.left_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=10)

        # Output of the detection limit formula
        self.result_label = tk.Label(self.left_frame, text="Equation:\n-", font=("Arial", 12),
                                     fg="black", justify=tk.CENTER)
        self.result_label.pack(fill=tk.X, padx=10, pady=10)

        # Right panel (graph)
        self.right_frame = tk.Frame(main_pane)
        main_pane.add(self.right_frame)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.right_frame)
        self.toolbar.update()
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def update_fwhm(self, *args):
        try:
            n_val = int(self.param_vars['N'].get())
            if n_val > 0:
                # Calculation of sigma (the width of the Gaussian base is 6 sigma)
                sigma = self.image_size_X_2 / (6 * n_val)
                fwhm = 2.355 * sigma
                self.fwhm_label.config(text=f"FWHM: {fwhm:.2f} px")
            else:
                self.fwhm_label.config(text="FWHM: -")
        except ValueError:
            self.fwhm_label.config(text="FWHM: -")

    def get_params(self):
        params = {}
        try:
            for key, var in self.param_vars.items():
                val_type = self.param_types[key][1]
                params[key] = val_type(float(var.get()))
            return params
        except ValueError as e:
            messagebox.showerror("Type Error", f"Invalid input for parameter: {key}")
            return None

    def start_calculation(self):
        params = self.get_params()
        if not params: return

        self.btn_limit.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        self.result_label.config(text="Equation:\n-")
        self.progress_var.set(0)
        self.select_mode = False
        self.btn_select.config(text="Select Data (OFF)")

        LAMBDA = self.CRVAL1 + self.CDELT1 * np.arange(self.start_idx, self.end_idx)
        Lambda1 = LAMBDA
        Lambda2 = LAMBDA

        Int1 = np.full(self.image_size_X_2, params['sp_level'])
        Int2 = np.full(self.image_size_X_2, params['I_cont'] * params['sp_level'])
        x = np.arange(self.image_size_X_2)

        peak_positions = []
        peak_amplitudes = []

        sp_level = params['sp_level']
        I_min_val = params['I_min'] * sp_level
        I_max_val = params['I_max'] * sp_level

        for i in range(params['N']):
            amp = I_min_val + (I_max_val - I_min_val) * i / (params['N'] - 1) if params['N'] > 1 else I_max_val
            mu = (i + 0.5) * self.image_size_X_2 / params['N']
            sigma = self.image_size_X_2 / (6 * params['N'])
            Int2 += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            peak_positions.append(mu)
            peak_amplitudes.append(amp / sp_level)

        seps = np.linspace(params['sep_min'], params['sep_max'], params['N_it'])
        all_obs_shifts = []

        try:
            for sst, sep in enumerate(seps):
                image_mod, PSF = Frame_model_known_spec(
                    params['slit'], params['seeing'], sep, params['pix_size'],
                    self.CRVAL1, self.image_size_X_2, self.image_size_Y_2,
                    self.Y_est_2, self.CDELT1, params['RN'], params['BN'],
                    Lambda1, Int1, Lambda2, Int2
                )

                accumulated = np.zeros_like(image_mod)

                for _ in range(params['N_obs']):
                    image_noisy = np.random.poisson(image_mod)
                    accumulated += image_noisy

                image_mod = accumulated / params['N_obs']

                OBS_LAMBDA, OBS_SPEC, OBS_CENTER, OBS_FWHM, OBS_ERRORBAR = Center_search(
                    LAMBDA[self.delt], self.CDELT1, self.delt, self.image_size_X_2,
                    params['area'], self.Y_est_2, image_mod
                )

                OBS_SHIFT = OBS_CENTER - np.mean(OBS_CENTER[0:5])
                all_obs_shifts.append(OBS_SHIFT)

                # Updating the progress bar
                self.progress_var.set((sst + 1) / params['N_it'] * 100)
                self.root.update()

        except Exception as e:
            messagebox.showerror("Execution Error", f"Error during calculation:\n{str(e)}")
            return

        shift_offset = np.max(all_obs_shifts) - np.min(all_obs_shifts)
        if shift_offset == 0:
            messagebox.showerror("Calculation Error", "Shift offset is zero.")
            return

        # Search for extremes
        noise_region = all_obs_shifts[0][:int(len(OBS_LAMBDA) * 0.2)]
        noise_std = np.std(noise_region)
        noise_threshold = params['threshold'] * noise_std

        first_noticeable_peaks_idx = []

        for i, shift in enumerate(all_obs_shifts):
            smoothed = gaussian_filter1d(shift, sigma=2.0)
            extrema_indices = []
            for j in range(2, len(smoothed) - 2):
                is_min = (smoothed[j] < smoothed[j - 1] and smoothed[j] < smoothed[j - 2] and
                          smoothed[j] < smoothed[j + 1] and smoothed[j] < smoothed[j + 2])
                is_max = (smoothed[j] > smoothed[j - 1] and smoothed[j] > smoothed[j - 2] and
                          smoothed[j] > smoothed[j + 1] and smoothed[j] > smoothed[j + 2])

                if is_min or is_max:
                    extrema_indices.append(j)

            sigma_range = self.image_size_X_2 / (6 * params['N'])
            found_first = False
            for ext_idx in extrema_indices:
                for k, peak_pos in enumerate(peak_positions):
                    peak_pos_obs = peak_pos - self.delt
                    if abs(ext_idx - peak_pos_obs) < sigma_range * 3:
                        if abs(smoothed[ext_idx] - np.mean(smoothed[:int(len(OBS_LAMBDA) * 0.2)])) > noise_threshold:
                            first_noticeable_peaks_idx.append(int(round(peak_pos_obs)))
                            found_first = True
                            break
                if found_first:
                    break

            if not found_first:
                first_noticeable_peaks_idx.append(None)

        shift_matrix = np.array(all_obs_shifts)

        peak_positions_obs = [p - self.delt for p in peak_positions if 0 <= p - self.delt < len(OBS_LAMBDA)]
        peak_amplitudes_obs = [peak_amplitudes[i] for i, p in enumerate(peak_positions) if
                               0 <= p - self.delt < len(OBS_LAMBDA)]

        # Saving session data
        self.session_data = {
            'params': params,
            'obs_lambda_len': len(OBS_LAMBDA),
            'shift_matrix': shift_matrix.tolist(),
            'seps': seps.tolist(),
            'first_idx': first_noticeable_peaks_idx,
            'peak_pos_obs': peak_positions_obs,
            'peak_amp_obs': peak_amplitudes_obs
        }

        self.btn_limit.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        self.draw_initial_plot()

    def draw_initial_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        d = self.session_data

        self.ax.imshow(
            np.array(d['shift_matrix']), aspect='auto', origin='lower',
            extent=[0, d['obs_lambda_len'], d['params']['sep_min'], d['params']['sep_max']], cmap='viridis'
        )

        self.ax.set_xlim(0, d['obs_lambda_len'])
        self.ax.set_ylim(d['params']['sep_min'], d['params']['sep_max'])

        self.points = []

        for i in range(len(d['seps'])):
            idx = d['first_idx'][i]
            if idx is not None and 0 <= idx <= d['obs_lambda_len']:
                distances = [abs(idx - p) for p in d['peak_pos_obs']]
                closest_k = np.argmin(distances)
                amp = d['peak_amp_obs'][closest_k]

                self.points.append({
                    'idx': idx, 'sep': d['seps'][i], 'amp': amp, 'selected': False
                })

        self.update_plot_points()

        self.ax.tick_params(axis='both', direction='in', size=0)
        self.ax.set_xticks(d['peak_pos_obs'])
        self.ax.set_xticklabels([f"{amp:.4f}" for amp in d['peak_amp_obs']])
        self.ax.set_xlabel(r"$\delta I$")
        self.ax.set_ylabel("sep, arcsec")
        self.fig.tight_layout()

        # Setting up a rectangular area selection tool with the mouse
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=False,
            button=[1],  # Left mouse button
            minspanx=0, minspany=0,
            spancoords='pixels'
        )
        self.selector.set_active(self.select_mode)

        self.canvas_plot.draw()

    def update_plot_points(self):
        for line in self.ax.lines:
            line.remove()

        # Drawing unselected points
        unsel_x = [p['idx'] for p in self.points if not p['selected']]
        unsel_y = [p['sep'] for p in self.points if not p['selected']]
        if unsel_x:
            self.ax.plot(unsel_x, unsel_y, 'ko', markersize=5)

        # Drawing selected points
        sel_x = [p['idx'] for p in self.points if p['selected']]
        sel_y = [p['sep'] for p in self.points if p['selected']]
        if sel_x:
            self.ax.plot(sel_x, sel_y, 'ro', markersize=6)

        self.canvas_plot.draw()

    def toggle_select(self):
        if self.select_mode:
            self.select_mode = False
            self.btn_select.config(text="Select Data (OFF)")
            if hasattr(self, 'selector'):
                self.selector.set_active(False)
            # Reset selections when you press it again
            for p in self.points:
                p['selected'] = False
            self.update_plot_points()
        else:
            self.select_mode = True
            self.btn_select.config(text="Select Data (ON)")
            if hasattr(self, 'selector'):
                self.selector.set_active(True)

    def on_select(self, eclick, erelease):
        if not self.select_mode or not self.points:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Checking for clicks outside the graph
        if None in (x1, y1, x2, y2):
            return

        # Tolerance (2% of the current axis scale)
        x_tol = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.02
        y_tol = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.02

        xmin, xmax = min(x1, x2) - x_tol, max(x1, x2) + x_tol
        ymin, ymax = min(y1, y2) - y_tol, max(y1, y2) + y_tol

        # Select all points inside the rectangle (including tolerance)
        for p in self.points:
            if xmin <= p['idx'] <= xmax and ymin <= p['sep'] <= ymax:
                p['selected'] = True

        self.update_plot_points()

        self.update_plot_points()

    def theor_limit(self, delta_I, a, b):
        return a / delta_I + b

    def find_limit(self):
        if not self.session_data or not self.points:
            return

        # Use only the selected points, if any (otherwise, use all of them)
        active_points = [p for p in self.points if p['selected']]
        if not active_points:
            active_points = self.points

        if len(active_points) < 2:
            messagebox.showerror("Fitting Error", "Not enough points for fitting. Need at least 2.")
            return

        x_data_amp = np.array([p['amp'] for p in active_points])
        y_data_sep = np.array([p['sep'] for p in active_points])

        try:
            popt, pcov = curve_fit(self.theor_limit, x_data_amp, y_data_sep)
        except Exception as e:
            messagebox.showerror("Fitting Error", f"Curve fitting failed:\n{str(e)}")
            return

        d = self.session_data

        # Redrawing points (so that the line is on top)
        self.update_plot_points()

        k_pix, b_pix = np.polyfit(d['peak_amp_obs'], d['peak_pos_obs'], 1)
        smooth_pix = np.linspace(0, d['obs_lambda_len'], 500)
        smooth_amp = (smooth_pix - b_pix) / k_pix

        valid_x_mask = np.abs(smooth_amp) > 1e-5
        smooth_pix = smooth_pix[valid_x_mask]
        smooth_amp = smooth_amp[valid_x_mask]

        smooth_sep = self.theor_limit(smooth_amp, *popt)
        valid_y_indices = (smooth_sep >= d['params']['sep_min']) & (smooth_sep <= d['params']['sep_max'])

        self.ax.plot(
            smooth_pix[valid_y_indices], smooth_sep[valid_y_indices], 'k--', linewidth=4
        )
        self.canvas_plot.draw()

        # Output of coefficients in the GUI
        if popt[1] > 0:
            eq_text = f"sep >= {popt[0]:.3f} / δI + {popt[1]:.3f}"
        else:
            eq_text = f"sep >= {popt[0]:.3f} / δI - {abs(popt[1]):.3f}"

        self.result_label.config(text=f"Equation:\n{eq_text}")

    def save_data(self):
        if not self.session_data:
            messagebox.showerror("Save Error", "No data to save. Please run 'Start' first.")
            return

        os.makedirs('data', exist_ok=True)
        filename = filedialog.asksaveasfilename(
            initialdir='data', defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], title="Save Session Data"
        )
        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Key", "Value"])

                # Saving parameters
                for k, v in self.session_data['params'].items():
                    writer.writerow([f"param_{k}", v])

                # Saving arrays
                writer.writerow(["obs_lambda_len", self.session_data['obs_lambda_len']])
                writer.writerow(["shift_matrix", json.dumps(self.session_data['shift_matrix'])])
                writer.writerow(["seps", json.dumps(self.session_data['seps'])])
                writer.writerow(["first_idx", json.dumps(self.session_data['first_idx'])])
                writer.writerow(["peak_pos_obs", json.dumps(self.session_data['peak_pos_obs'])])
                writer.writerow(["peak_amp_obs", json.dumps(self.session_data['peak_amp_obs'])])

                # Saving the state of selected points
                points_state = [{'idx': p['idx'], 'selected': p['selected']} for p in self.points]
                writer.writerow(["points_state", json.dumps(points_state)])

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data:\n{str(e)}")

    def load_data(self):
        filename = filedialog.askopenfilename(
            initialdir='data', filetypes=[("CSV files", "*.csv")], title="Load Session Data"
        )
        if not filename:
            return

        try:
            loaded_data = {'params': {}}
            points_state = []
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) != 2: continue
                    key, val = row[0], row[1]

                    if key.startswith("param_"):
                        p_name = key.replace("param_", "")
                        if p_name in self.param_types:
                            val_type = self.param_types[p_name][1]
                            loaded_data['params'][p_name] = val_type(float(val))
                            self.param_vars[p_name].set(str(loaded_data['params'][p_name]))
                    elif key == "obs_lambda_len":
                        loaded_data['obs_lambda_len'] = int(val)
                    elif key in ["shift_matrix", "seps", "first_idx", "peak_pos_obs", "peak_amp_obs"]:
                        loaded_data[key] = json.loads(val)
                    elif key == "points_state":
                        points_state = json.loads(val)

            self.session_data = loaded_data
            self.btn_limit.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.NORMAL)

            # Drawing a graph
            self.draw_initial_plot()

            # Restoring selected points
            if points_state:
                state_dict = {p['idx']: p['selected'] for p in points_state}
                for p in self.points:
                    if p['idx'] in state_dict:
                        p['selected'] = state_dict[p['idx']]
                self.update_plot_points()

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load data:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TalosApp(root)
    root.mainloop()
