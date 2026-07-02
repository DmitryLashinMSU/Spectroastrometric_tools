import sys
import tkinter as tk

if __name__ == '__main__':
    # If a child process is running
    if len(sys.argv) > 1:
        if sys.argv[1] == "processing":
            import SPARTA_processing

            root = tk.Tk()
            app = SPARTA_processing.SAHeatmapsApp(root)
            root.mainloop()
            sys.exit(0)

        elif sys.argv[1] == "separation":
            import SPARTA_separation

            root = tk.Tk()
            app = SPARTA_separation.SpartaApp(root)
            root.mainloop()
            sys.exit(0)

    # Default launch
    import SPARTA_preprocessing

    app = SPARTA_preprocessing.SpectroAstrometryApp()
    app.mainloop()