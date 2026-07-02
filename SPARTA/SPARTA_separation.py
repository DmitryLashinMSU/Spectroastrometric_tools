import os
import sys
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class SpartaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPARTA (Spectral Processing for Analysis, Reduction and Two-source Astrometry)")

        # Setting the window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.35)
        window_height = int(screen_height * 0.6)
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(400, 400)

        # Window closing processing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables for storing source data
        self.default_params = {}
        self.wavelength = None
        self.flux_pa1 = None
        self.flux_pa2 = None
        self.sa_signal = None
        self.sa_err = None
        self.med_sm_val = 1

        # A list for storing vertical line objects
        self.vlines = []

        # Variables for storing the current calculated spectra
        self.norm_averaged_flux = None
        self.flux_source1 = None
        self.flux_source2 = None
        self.err_source1 = None
        self.err_source2 = None

        self.load_data()
        self.init_ui()
        self.set_defaults()

    def load_data(self):
        d1, d1_err, d2, d2_err = [None] * 4

        try:
            with open('data/processing_data.csv', 'r') as f:
                for line in f:
                    # The Sep field is ignored if it remains in the file
                    if line.startswith('d1,'): d1 = float(line.split(',')[1])
                    if line.startswith('d1_Err,'): d1_err = float(line.split(',')[1])
                    if line.startswith('d2,'): d2 = float(line.split(',')[1])
                    if line.startswith('d2_Err,'): d2_err = float(line.split(',')[1])
        except FileNotFoundError:
            messagebox.showerror("Error", "File data/processing_data.csv not found!")
            sys.exit()

        self.default_params = {
            'd1': d1, 'd1_err': d1_err,
            'd2': d2, 'd2_err': d2_err,
            'med_sm': 1  # Default value for smoothing
        }

        wl_start, wl_end = None, None
        try:
            with open('data/params_and_data.csv', 'r') as f:
                for line in f:
                    if not line.startswith('#'): break
                    if 'wl_start:' in line: wl_start = float(line.split(':')[1].strip())
                    if 'wl_end:' in line: wl_end = float(line.split(':')[1].strip())

            data = pd.read_csv('data/params_and_data.csv', comment='#')
            if wl_start is not None and wl_end is not None:
                mask = (data['Wavelength'] >= wl_start) & (data['Wavelength'] <= wl_end)
                data = data[mask]

            self.wavelength = data['Wavelength'].values
            self.flux_pa1 = data['Flux_PA1'].values
            self.flux_pa2 = data['Flux_PA2'].values
            self.sa_signal = data['Half_Diff'].values
            self.sa_err = data['Err_Half_Diff'].values

        except FileNotFoundError:
            messagebox.showerror("Error", "File data/params_and_data.csv not found!")
            sys.exit()

    def init_ui(self):
        # Top panel with scrolling and centring
        self.top_container = tk.Frame(self.root)
        self.top_container.pack(side=tk.TOP, fill=tk.X)

        self.scrollbar = tk.Scrollbar(self.top_container, orient="horizontal")
        self.scrollbar_visible = False

        self.canvas = tk.Canvas(self.top_container, height=120, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.scrollbar.configure(command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.tools_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.tools_frame, anchor="nw")

        # Frame for input fields (automatically centered using pack)
        entries_frame = tk.Frame(self.tools_frame)
        entries_frame.pack(side=tk.TOP, pady=(10, 5))

        self.entries = {}

        # Grouping of parameters
        param_groups = [
            ('d1', 'd1', 'd1_err'),
            ('d2', 'd2', 'd2_err')
        ]

        col = 0
        font_style = ("Arial", 10)

        for display_name, val_key, err_key in param_groups:
            # Name of the parameter
            tk.Label(entries_frame, text=f"{display_name} =", font=font_style).grid(row=0, column=col, padx=(5, 2),
                                                                                    pady=5)
            col += 1

            # Value field
            ent_val = tk.Entry(entries_frame, width=8)
            ent_val.grid(row=0, column=col, padx=2, pady=5)
            self.entries[val_key] = ent_val
            col += 1

            # ± sign
            tk.Label(entries_frame, text="±", font=font_style).grid(row=0, column=col, padx=2, pady=5)
            col += 1

            # Field of error
            ent_err = tk.Entry(entries_frame, width=8)
            ent_err.grid(row=0, column=col, padx=2, pady=5)
            self.entries[err_key] = ent_err
            col += 1

            # Units (pix) and spacing between groups
            tk.Label(entries_frame, text="pix", font=font_style).grid(row=0, column=col, padx=(2, 25), pady=5)
            col += 1

        # Field for the med_sm smoothing parameter
        tk.Label(entries_frame, text="med_sm =", font=font_style).grid(row=0, column=col, padx=(5, 2), pady=5)
        col += 1

        ent_med = tk.Entry(entries_frame, width=5)
        ent_med.grid(row=0, column=col, padx=2, pady=5)
        self.entries['med_sm'] = ent_med
        col += 1

        tk.Label(entries_frame, text="pix", font=font_style).grid(row=0, column=col, padx=(2, 5), pady=5)

        # Frame for buttons (automatically centered using pack)
        btn_frame = tk.Frame(self.tools_frame)
        btn_frame.pack(side=tk.TOP, pady=(10, 200))

        tk.Button(btn_frame, text="Recount", bg="gray80", command=self.recount).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Set Defaults", bg="gray80", command=self.set_defaults).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Save Results", bg="gray80", command=self.save_results).pack(side=tk.LEFT, padx=10)

        # Event binding for dynamic resizing
        self.tools_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Bottom panel with graphs
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.fig.subplots_adjust(hspace=0.1, left=0.1, right=0.95, top=0.95, bottom=0.1)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Enabling the mouse click event
        self.canvas_plot.mpl_connect('button_press_event', self.on_plot_click)

        toolbar = NavigationToolbar2Tk(self.canvas_plot, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # Right-click handle
    def on_plot_click(self, event):
        if event.button == 3 and event.inaxes:  #3 - right mouse button
            # Removing old lines
            for line in self.vlines:
                line.remove()
            self.vlines.clear()

            # Drawing new lines on all 3 pictures
            for ax in self.axs:
                line = ax.axvline(x=event.xdata, color='green', linestyle='--')
                self.vlines.append(line)

            self.canvas_plot.draw()

    # Internal frame rescaling event
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._check_scrollbar()

    # External Canvas rescaling event
    def _on_canvas_configure(self, event=None):
        self._check_scrollbar()

    def _check_scrollbar(self):
        canvas_width = self.canvas.winfo_width()
        req_width = self.tools_frame.winfo_reqwidth()

        if canvas_width <= 1:
            return  # The window has not been fully rendered yet

        # Centering elements inside the tools_frame
        new_width = max(canvas_width, req_width)
        self.canvas.itemconfig(self.canvas_window, width=new_width)

        # Scrollbar display logic
        if req_width > canvas_width:
            if not self.scrollbar_visible:
                self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                self.scrollbar_visible = True
        else:
            if self.scrollbar_visible:
                self.scrollbar.pack_forget()
                self.scrollbar_visible = False

    def set_defaults(self):
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(self.default_params[key]))
        self.recount()

    def recount(self):
        try:
            d1 = float(self.entries['d1'].get())
            d1_err = float(self.entries['d1_err'].get())
            d2 = float(self.entries['d2'].get())
            d2_err = float(self.entries['d2_err'].get())
            self.med_sm_val = int(self.entries['med_sm'].get())
            if self.med_sm_val < 1:
                raise ValueError("med_sm must be >= 1")
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Check the correctness of the entered numeric values! (med_sm must be an integer >= 1)")
            return

        sep = abs(d1 - d2)

        # Protection against division by zero
        if sep == 0:
            messagebox.showerror("Math Error", "Parameters d1 and d2 cannot be equal!")
            return

        # Initial normalized averaged spectrum
        norm_pa1 = self.flux_pa1 / max(self.flux_pa1)
        norm_pa2 = self.flux_pa2 / max(self.flux_pa2)
        self.norm_averaged_flux = (norm_pa1 + norm_pa2) / 2.0

        # Calculating errors
        term_sa = (self.sa_err / sep) ** 2
        term_d1 = (((self.sa_signal - d2) / sep ** 2) * d1_err) ** 2
        term_d2 = (((self.sa_signal - d1) / sep ** 2) * d2_err) ** 2

        f_err = np.sqrt(term_sa + term_d1 + term_d2)

        # Total error for the spectrum
        norm_flux_err = self.norm_averaged_flux * f_err

        # Correlation of reconstructed spectra and sources
        norm_flux_A = self.norm_averaged_flux * (self.sa_signal - max(d1, d2)) / sep
        norm_flux_B = self.norm_averaged_flux - norm_flux_A

        if d1 > d2:
            # If d1 > d2, then norm_flux_A refers to the "second" source
            self.flux_source1 = norm_flux_B
            self.flux_source2 = norm_flux_A
        else:
            # Otherwise, go to the first one
            self.flux_source1 = norm_flux_A
            self.flux_source2 = norm_flux_B

        self.err_source1 = norm_flux_err
        self.err_source2 = norm_flux_err

        # Median smoothing
        if self.med_sm_val > 1:
            def apply_smoothing(data_array):
                # pandas rolling median. min_periods=1 prevents edge clipping (NaN)
                return pd.Series(data_array).rolling(window=self.med_sm_val, center=True, min_periods=1).median().values

            self.norm_averaged_flux = apply_smoothing(self.norm_averaged_flux)
            self.flux_source1 = apply_smoothing(self.flux_source1)
            self.flux_source2 = apply_smoothing(self.flux_source2)
            self.err_source1 = apply_smoothing(self.err_source1)
            self.err_source2 = apply_smoothing(self.err_source2)

        self.update_plots()

    def update_plots(self):
        for ax in self.axs:
            ax.clear()
        self.vlines = []  # Resetting the line list after clearing the axes

        wl = self.wavelength

        # Initial averaged normalized spectrum
        self.axs[0].plot(wl, self.norm_averaged_flux, color='black', label='Averaged Normalized Flux')
        self.axs[0].set_ylabel('Norm. Flux')
        self.axs[0].grid(True, alpha=0.3)

        # Reconstructed spectrum of Source 1
        self.axs[1].plot(wl, self.flux_source1, color='C0', label='Source 1')
        self.axs[1].fill_between(wl, self.flux_source1 - self.err_source1,
                                 self.flux_source1 + self.err_source1, color='C0', alpha=0.3)
        self.axs[1].set_ylabel('Flux (Source 1)')
        self.axs[1].grid(True, alpha=0.3)

        # Reconstructed spectrum of Source 2
        self.axs[2].plot(wl, self.flux_source2, color='C1', label='Source 2')
        self.axs[2].fill_between(wl, self.flux_source2 - self.err_source2,
                                 self.flux_source2 + self.err_source2, color='C1', alpha=0.3)
        self.axs[2].set_ylabel('Flux (Source 2)')
        self.axs[2].set_xlabel('Wavelength')
        self.axs[2].grid(True, alpha=0.3)

        self.canvas_plot.draw()

    def save_results(self):
        if self.flux_source1 is None:
            messagebox.showwarning("Warning", "No data to save! Press Recount.")
            return

        out_dir = 'data'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, 'separated_spectra.csv')

        df_out = pd.DataFrame({
            'Wavelength': self.wavelength,
            'Averaged_Norm_Flux': self.norm_averaged_flux,
            'Source1_Flux': self.flux_source1,
            'Source1_Err': self.err_source1,
            'Source2_Flux': self.flux_source2,
            'Source2_Err': self.err_source2
        })

        try:
            df_out.to_csv(out_path, index=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        sys.exit()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpartaApp(root)
    root.mainloop()
