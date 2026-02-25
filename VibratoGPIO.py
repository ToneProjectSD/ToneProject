import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import sounddevice as sd
from collections import deque
import json
import os


EPSILON = 1e-12


def _parabolic_peak(values, center_idx):
    if center_idx <= 0 or center_idx >= len(values) - 1:
        return float(center_idx)

    left = values[center_idx - 1]
    center = values[center_idx]
    right = values[center_idx + 1]
    denom = left - (2.0 * center) + right
    if abs(denom) < EPSILON:
        return float(center_idx)

    delta = 0.5 * (left - right) / denom
    if abs(delta) > 1.0:
        return float(center_idx)
    return float(center_idx + delta)


def _moving_average(values, window_size):
    if len(values) == 0 or window_size <= 1 or len(values) < window_size:
        return values.copy()
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def smooth_and_filter_pitch_track(raw_freqs):
    if len(raw_freqs) < 5:
        return np.array(raw_freqs, dtype=np.float64)

    smoothed = np.array([
        np.median(raw_freqs[max(0, i - 2):min(len(raw_freqs), i + 3)])
        for i in range(len(raw_freqs))
    ], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(np.maximum(smoothed, EPSILON))

    if len(cents) >= 16:
        bin_width = 25.0
        bins = np.arange(np.min(cents) - bin_width, np.max(cents) + (2.0 * bin_width), bin_width)
        hist, edges = np.histogram(cents, bins=bins)
        dominant_idx = int(np.argmax(hist))
        cluster_center = 0.5 * (edges[dominant_idx] + edges[dominant_idx + 1])
        
        diffs = cents - cluster_center
        clipped_diffs = np.clip(diffs, -150.0, 150.0)
        cleaned_cents = cluster_center + clipped_diffs
        
        return 2.0 ** (cleaned_cents / 1200.0)

    return smoothed


def estimate_modulation_rate_hz(cents_track, sample_rate_hz, min_rate_hz=2.0, max_rate_hz=12.0):
    if len(cents_track) < 20:
        return 0.0, 0.0

    detrended = cents_track - np.mean(cents_track)
    
    acf = np.correlate(detrended, detrended, mode='full')
    acf = acf[len(acf)//2:]
    if acf[0] < EPSILON:
        return 0.0, 0.0
        
    acf = acf / acf[0]
    
    min_period = int(sample_rate_hz / max_rate_hz)
    max_period = int(sample_rate_hz / min_rate_hz)
    if max_period >= len(acf): 
        max_period = len(acf) - 1
    if min_period >= max_period: 
        return 0.0, 0.0
    
    valid_acf = acf[min_period:max_period+1]
    if len(valid_acf) == 0: 
        return 0.0, 0.0
    
    peak_idx = int(np.argmax(valid_acf))
    best_lag = min_period + peak_idx
    confidence = float(valid_acf[peak_idx])
    
    refined_lag = _parabolic_peak(acf, best_lag)
    rate_hz = sample_rate_hz / refined_lag
    
    rate_hz = float(max(min_rate_hz, min(max_rate_hz, rate_hz)))
    return rate_hz, confidence


def estimate_fundamental_frequency(audio_window, rate, fft_size, min_f0, max_f0, expected_f0=None, prev_f0=None):
    t_min = int(rate / max_f0)
    t_max = int(rate / min_f0)
    N = len(audio_window)
    
    y_pad = np.pad(audio_window, (0, N))
    Y = np.fft.rfft(y_pad)
    acf = np.fft.irfft(Y * np.conjugate(Y))[:N]
    
    sq = audio_window ** 2
    cumsum_sq = np.concatenate(([0], np.cumsum(sq)))
    
    t_arr = np.arange(1, t_max + 1)
    e1 = cumsum_sq[N - t_arr]
    e2 = cumsum_sq[N] - cumsum_sq[t_arr]
    
    d = np.zeros(t_max + 1)
    d[1:t_max + 1] = e1 + e2 - 2 * acf[1:t_max + 1]
    
    running_sum = np.cumsum(d[1:t_max + 1])
    cmnd = np.zeros(t_max + 1)
    cmnd[0] = 1.0
    cmnd[1:t_max + 1] = d[1:t_max + 1] * t_arr / (running_sum + EPSILON)
    
    threshold = 0.15
    below_thresh = np.where(cmnd[t_min:t_max + 1] < threshold)[0]
    
    tau = None
    if len(below_thresh) > 0:
        tau = t_min + below_thresh[0]
        while tau + 1 <= t_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = t_min + np.argmin(cmnd[t_min:t_max + 1])
        if cmnd[tau] > 0.4:  
            return None, 0.0
            
    refined_tau = _parabolic_peak(cmnd, tau)
    f0 = rate / refined_tau
    confidence = 1.0 - cmnd[tau]
    
    return float(f0), float(confidence)


class VibratoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vibrato Project - Team T.O.N.E.")

        self.attributes('-fullscreen', True)
        self.geometry("480x320")
        self.configure(bg="#222222") # Dark mode base for embedded feel
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.SETTINGS_FILE = "vibrato_settings.json"

        # --- Find the I2S Microphone ---
        self.audio_device_index = self.find_i2s_device()

        # Tweak colors for dark mode UI
        self.THEMES = {
            "Standard": {
                "good": "#4CAF50", "good_bg": "#1b5e20",
                "warn": "#FFC107", "warn_bg": "#f57f17",
                "bad": "#F44336", "bad_bg": "#b71c1c",
                "span": "#FFFFFF", "bg": "#222222", "fg": "#FFFFFF"
            },
            "Protanopia/Deuteranopia": {
                "good": "#0072B2", "good_bg": "#004a75",
                "warn": "#E69F00", "warn_bg": "#996a00",
                "bad": "#D55E00", "bad_bg": "#8c3e00",
                "span": "#FFFFFF", "bg": "#222222", "fg": "#FFFFFF"
            },
            "Tritanopia": {
                "good": "#009E73", "good_bg": "#00664b",
                "warn": "#F0E442", "warn_bg": "#b3aa2c",
                "bad": "#D55E00", "bad_bg": "#8c3e00",
                "span": "#FFFFFF", "bg": "#222222", "fg": "#FFFFFF"
            },
            "High Contrast": {
                "good": "#FFFFFF", "good_bg": "#666666",
                "warn": "#AAAAAA", "warn_bg": "#444444",
                "bad": "#FF0000", "bad_bg": "#880000",
                "span": "#FFFFFF", "bg": "#000000", "fg": "#FFFFFF"
            }
        }

        self.current_theme = self.load_settings()

        # UI Styling optimized for embedded device (Flat, Chunky)
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        self.apply_theme_styles()

        self.container = ttk.Frame(self, style="Main.TFrame")
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, VibratoPage, TunerPage, SettingsPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def find_i2s_device(self):
        """
        Scans audio devices to find the I2S microphone (INMP441).
        It looks for keywords usually associated with the I2S overlay.
        """
        print("Scanning for Audio Devices...")
        devices = sd.query_devices()
        target_index = None
        
        # Common names for I2S overlays on Pi
        keywords = ['i2s', 'google', 'voicehat', 'dmic', 'simple-card']

        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (In: {dev['max_input_channels']})")
            # We need an input device
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                for k in keywords:
                    if k in name:
                        target_index = i
                        break
        
        if target_index is not None:
            print(f"--> Selected Device Index: {target_index} ({devices[target_index]['name']})")
            return target_index
        else:
            # Fallback: try to find *any* input device if specific one not found
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"--> I2S not found, defaulting to: {i} ({dev['name']})")
                    return i
            
            print("--> NO INPUT DEVICE FOUND!")
            return None

    def apply_theme_styles(self):
        bg_col = self.THEMES[self.current_theme]["bg"]
        fg_col = self.THEMES[self.current_theme]["fg"]
        
        self.style.configure("Main.TFrame", background=bg_col)
        self.style.configure("TLabel", background=bg_col, foreground=fg_col)
        
        # Massive buttons for physical knob navigation
        self.style.configure("Menu.TButton", font=("Helvetica", 18, "bold"), padding=15)
        self.style.configure("Header.TButton", font=("Helvetica", 11, "bold"), padding=8)
        self.style.configure("Action.TButton", font=("Helvetica", 14, "bold"), padding=10)

    def load_settings(self):
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    theme = data.get("theme", "Standard")
                    if theme in self.THEMES:
                        return theme
        except Exception:
            pass
        return "Standard"

    def save_settings(self):
        try:
            data = {"theme": self.current_theme}
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def get_color(self, role):
        return self.THEMES[self.current_theme].get(role, "#FFFFFF")

    def cycle_theme(self):
        keys = list(self.THEMES.keys())
        current_idx = keys.index(self.current_theme)
        next_idx = (current_idx + 1) % len(keys)
        self.current_theme = keys[next_idx]
        self.save_settings()
        self.apply_theme_styles()
        
        # Refresh current frame
        for frame in self.frames.values():
            if hasattr(frame, "on_show"):
                frame.on_show()
                
        return self.current_theme


class StartPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="Main.TFrame")
        self.controller = controller

        # Full width layout, perfect for scrolling with a knob
        lbl = ttk.Label(self, text="T.O.N.E. DEVICE", font=("Helvetica", 18, "bold"))
        lbl.pack(pady=(15, 10))

        # Show detected device on start screen for debugging
        dev_name = "No Mic Found"
        if controller.audio_device_index is not None:
            try:
                dev_info = sd.query_devices(controller.audio_device_index)
                dev_name = dev_info['name']
                if len(dev_name) > 20: dev_name = dev_name[:20] + "..."
            except:
                pass
        
        ttk.Label(self, text=f"Mic: {dev_name}", font=("Helvetica", 10)).pack(pady=(0, 10))

        btn_frame = ttk.Frame(self, style="Main.TFrame")
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))

        buttons = [
            ("Vibrato Training", "VibratoPage"),
            ("Tuner", "TunerPage"),
            ("Settings", "SettingsPage")
        ]

        for text, page in buttons:
            btn = ttk.Button(btn_frame, text=text, style="Menu.TButton",
                             command=lambda p=page: controller.show_frame(p))
            btn.pack(fill=tk.X, pady=5)


class VibratoPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="Main.TFrame")
        self.controller = controller

        self.is_analyzing = False
        self.analysis_thread = None
        
        # UPDATED: I2S mics prefer 48000Hz on RPi
        self.RATE = 48000
        self.WINDOW_SIZE = 2048
        self.STEP_SIZE = 512 
        self.FFT_SIZE = 8192
        self.MIN_F0 = 180.0
        self.MAX_F0_VIOLIN = 760.0
        self.MAX_F0_CHROMATIC = 1200.0
        self.SEGMENT_RESET_JUMP_CENTS = 220.0
        self.MAX_VALID_CENTS = 120.0
        self.MAX_DEPTH_CENTS = 100.0
        
        self.VIBRATO_MIN_DEPTH_CENTS = 12.0
        self.VIBRATO_MIN_RATE_HZ = 2.4
        self.VIBRATO_MAX_RATE_HZ = 9.5
        self.VIBRATO_MIN_CONFIDENCE = 0.2
        self.VIBRATO_HOLD_SECONDS = 1.1
        
        self.DISPLAY_RATE_ALPHA = 0.15
        self.DISPLAY_DEPTH_ALPHA = 0.15
        self.DISPLAY_DECAY = 0.88

        self.pitch_lock = threading.Lock()
        self.is_processing_stats = False

        self.pitch_history = deque(maxlen=130)
        self.audio_buffer = np.zeros(self.WINDOW_SIZE)
        self.prev_pitch_hz = None
        self.jump_counter = 0

        self.rate_display_history = deque(maxlen=6)
        self.depth_display_history = deque(maxlen=6)
        self.last_display_rate_hz = None
        self.last_display_depth_cents = None
        self.last_vibrato_timestamp = 0.0

        self.is_violin_mode = True
        self.VIOLIN_NOTES = {"G3": 196.00, "D4": 293.66, "A4": 440.00, "E5": 659.25}

        self.target_note_key = "Auto"
        self.violin_target_list = ["Auto", "G3", "D4", "A4", "E5"]
        self.violin_target_idx = 0

        self.CHROMATIC_NOTES = {}
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for midi in range(48, 85):
            note_idx = midi % 12
            octave = (midi // 12) - 1
            name = note_names[note_idx]
            freq = 440 * 2 ** ((midi - 69) / 12.0)
            self.CHROMATIC_NOTES[f"{name}{octave}"] = freq

        # --- UI LAYOUT FOR 3.2" SCREEN ---
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=2, padx=2)

        ttk.Button(header, text="< Menu", style="Header.TButton", command=self.go_back).pack(side=tk.LEFT, padx=2)
        
        self.btn_analyze = ttk.Button(header, text="START", style="Header.TButton", command=self.toggle_analysis)
        self.btn_analyze.pack(side=tk.RIGHT, padx=2)

        self.btn_mode = ttk.Button(header, text="Violin", style="Header.TButton", command=self.toggle_mode)
        self.btn_mode.pack(side=tk.RIGHT, padx=2)

        vis_frame = ttk.Frame(self, style="Main.TFrame")
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        vis_frame.columnconfigure(0, weight=2)
        vis_frame.columnconfigure(1, weight=3)
        vis_frame.columnconfigure(2, weight=2)
        vis_frame.rowconfigure(0, weight=1)

        # Rate Gauge Column
        col1 = ttk.Frame(vis_frame, style="Main.TFrame")
        col1.grid(row=0, column=0, sticky="nsew")
        ttk.Label(col1, text="RATE", font=("Helvetica", 10, "bold")).pack(side=tk.TOP)
        self.lbl_rate_val = ttk.Label(col1, text="-- Hz", font=("Helvetica", 16, "bold"))
        self.lbl_rate_val.pack(side=tk.TOP)

        self.gauge_w = 60
        self.canvas_rate = tk.Canvas(col1, width=self.gauge_w, bg=self.controller.THEMES[self.controller.current_theme]["bg"], highlightthickness=0)
        self.canvas_rate.pack(fill=tk.BOTH, expand=True, pady=2)

        # Center Column (Note)
        col2 = ttk.Frame(vis_frame, style="Main.TFrame")
        col2.grid(row=0, column=1, sticky="nsew")
        note_container = ttk.Frame(col2, style="Main.TFrame")
        note_container.pack(expand=True)
        
        self.lbl_note = ttk.Label(note_container, text="--", font=("Helvetica", 48, "bold"))
        self.lbl_note.pack()
        self.lbl_freq = ttk.Label(note_container, text="-- Hz", font=("Helvetica", 12))
        self.lbl_freq.pack(pady=(0,10))

        self.btn_target = ttk.Button(note_container, text="Tgt: Auto", style="Header.TButton", command=self.handle_target_click)
        self.btn_target.pack(pady=5)
        
        self.btn_dsp = ttk.Button(note_container, text="DSP Tune", style="Header.TButton", command=self.open_dsp_settings)
        self.btn_dsp.pack(pady=5)

        # Depth Gauge Column
        col3 = ttk.Frame(vis_frame, style="Main.TFrame")
        col3.grid(row=0, column=2, sticky="nsew")
        ttk.Label(col3, text="DEPTH", font=("Helvetica", 10, "bold")).pack(side=tk.TOP)
        self.lbl_depth_val = ttk.Label(col3, text="-- ct", font=("Helvetica", 16, "bold"))
        self.lbl_depth_val.pack(side=tk.TOP)

        self.canvas_depth = tk.Canvas(col3, width=self.gauge_w, bg=self.controller.THEMES[self.controller.current_theme]["bg"], highlightthickness=0)
        self.canvas_depth.pack(fill=tk.BOTH, expand=True, pady=2)

        self.canvas_rate.bind("<Configure>", lambda e: self.redraw_gauges())
        self.redraw_gauges()

    def open_dsp_settings(self):
        top = tk.Toplevel(self)
        top.title("DSP Setup")
        top.geometry("320x240")
        top.configure(bg=self.controller.THEMES[self.controller.current_theme]["bg"])
        
        # Center tightly
        x = self.winfo_x() + 80
        y = self.winfo_y() + 40
        top.geometry(f"+{x}+{y}")

        ttk.Label(top, text="Win Size:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        var_win = tk.StringVar(value=str(self.WINDOW_SIZE))
        ttk.Combobox(top, textvariable=var_win, values=["1024", "2048", "4096"], state="readonly", width=8).grid(row=0, column=1, pady=5)

        ttk.Label(top, text="Step Size:", font=("Helvetica", 10, "bold")).grid(row=1, column=0, padx=5, pady=5, sticky="e")
        var_step = tk.StringVar(value=str(self.STEP_SIZE))
        ttk.Combobox(top, textvariable=var_step, values=["256", "512", "1024"], state="readonly", width=8).grid(row=1, column=1, pady=5)

        ttk.Label(top, text="History:", font=("Helvetica", 10, "bold")).grid(row=2, column=0, padx=5, pady=5, sticky="e")
        var_hist = tk.IntVar(value=self.pitch_history.maxlen)
        ttk.Scale(top, from_=40, to=200, variable=var_hist, orient=tk.HORIZONTAL, length=100).grid(row=2, column=1, pady=5)

        def apply_and_close():
            was_analyzing = self.is_analyzing
            if was_analyzing: self.toggle_analysis() 

            self.WINDOW_SIZE = int(var_win.get())
            self.STEP_SIZE = int(var_step.get())
            
            with self.pitch_lock:
                self.pitch_history = deque(maxlen=var_hist.get())
                self.audio_buffer = np.zeros(self.WINDOW_SIZE)

            if was_analyzing: self.toggle_analysis() 
            top.destroy()

        ttk.Button(top, text="Save & Close", style="Action.TButton", command=apply_and_close).grid(row=3, column=0, columnspan=2, pady=15)

    def handle_target_click(self):
        if self.is_violin_mode:
            self.violin_target_idx = (self.violin_target_idx + 1) % len(self.violin_target_list)
            self.target_note_key = self.violin_target_list[self.violin_target_idx]
            self.btn_target.config(text=f"Tgt: {self.target_note_key}")
        else:
            self.open_chromatic_popup()

    def open_chromatic_popup(self):
        top = tk.Toplevel(self)
        top.title("Target")
        x = self.winfo_x() + 40
        y = self.winfo_y() + 20
        top.geometry(f"400x280+{x}+{y}")
        top.configure(bg=self.controller.THEMES[self.controller.current_theme]["bg"])

        ttk.Label(top, text="Select Note", font=("Helvetica", 14, "bold")).pack(pady=5)

        self.sel_octave = tk.StringVar(value="4")
        frm_oct = ttk.Frame(top, style="Main.TFrame")
        frm_oct.pack(pady=5)
        for oct_val in ["3", "4", "5"]:
            ttk.Radiobutton(frm_oct, text=f"Oct {oct_val}", variable=self.sel_octave, value=oct_val).pack(side=tk.LEFT, padx=10)

        frm_grid = ttk.Frame(top, style="Main.TFrame")
        frm_grid.pack(pady=5)
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def set_chrom_target(n):
            octave = self.sel_octave.get()
            final_note = f"{n}{octave}"
            self.target_note_key = final_note
            self.btn_target.config(text=f"Tgt: {final_note}")
            top.destroy()

        r, c = 0, 0
        for n in notes:
            btn = ttk.Button(frm_grid, text=n, width=4, style="Header.TButton", command=lambda x=n: set_chrom_target(x))
            btn.grid(row=r, column=c, padx=5, pady=5)
            c += 1
            if c > 3: c = 0; r += 1

        frm_bott = ttk.Frame(top, style="Main.TFrame")
        frm_bott.pack(pady=10)
        ttk.Button(frm_bott, text="Auto Mode", style="Action.TButton", command=lambda: [self.set_auto_chromatic(), top.destroy()]).pack(side=tk.LEFT, padx=10)

    def set_auto_chromatic(self):
        self.target_note_key = "Auto"
        self.btn_target.config(text="Tgt: Auto")

    def go_back(self):
        if self.is_analyzing:
            self.is_analyzing = False
            self.btn_analyze.config(text="START")
            self.reset_display()
        self.controller.show_frame("StartPage")

    def on_show(self):
        self.canvas_rate.configure(bg=self.controller.THEMES[self.controller.current_theme]["bg"])
        self.canvas_depth.configure(bg=self.controller.THEMES[self.controller.current_theme]["bg"])
        self.redraw_gauges()

    def redraw_gauges(self):
        # Clean Embedded style gauges - thick track
        self.draw_embedded_gauge(self.canvas_rate, 0, 12, 3, 9, 2, 10, symmetric=False)
        self.draw_embedded_gauge(self.canvas_depth, -50, 50, 5, 20, 0, 40, symmetric=True)

    def draw_embedded_gauge(self, canvas, min_val, max_val, target_min, target_max, warn_min, warn_max, symmetric=False):
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h_draw = canvas.winfo_height()
        if w < 10: w = 60
        if h_draw < 10: h_draw = 180

        canvas.delete("all")
        
        c_good = self.controller.get_color("good_bg")
        c_warn = self.controller.get_color("warn_bg")
        c_bad = self.controller.get_color("bad_bg")

        bar_w = int(w * 0.5)
        bar_x1 = int((w - bar_w) / 2)
        bar_x2 = bar_x1 + bar_w

        def val_to_y(v):
            pct = (v - min_val) / (max_val - min_val)
            pct = max(0.0, min(1.0, pct))
            return h_draw - (pct * h_draw)

        # Draw segmented track
        if not symmetric:
            canvas.create_rectangle(bar_x1, val_to_y(max_val), bar_x2, val_to_y(min_val), fill=c_bad, outline="")
            canvas.create_rectangle(bar_x1, val_to_y(warn_max), bar_x2, val_to_y(warn_min), fill=c_warn, outline="")
            canvas.create_rectangle(bar_x1, val_to_y(target_max), bar_x2, val_to_y(target_min), fill=c_good, outline="")
        else:
            canvas.create_rectangle(bar_x1, val_to_y(max_val), bar_x2, val_to_y(min_val), fill=c_bad, outline="")
            canvas.create_rectangle(bar_x1, val_to_y(warn_max), bar_x2, val_to_y(-warn_max), fill=c_warn, outline="")
            canvas.create_rectangle(bar_x1, val_to_y(target_max), bar_x2, val_to_y(-target_max), fill=c_good, outline="")
            
            y_zero = val_to_y(0)
            canvas.create_line(bar_x1 - 5, y_zero, bar_x2 + 5, y_zero, fill=self.controller.get_color("fg"), width=2)

        canvas.gauge_meta = {'min': min_val, 'max': max_val, 'h_draw': h_draw, 'bar_x1': bar_x1, 'bar_x2': bar_x2, 'symmetric': symmetric}

    def update_gauge_needle(self, canvas, value, val_b=None):
        if not hasattr(canvas, 'gauge_meta'): return
        m = canvas.gauge_meta
        canvas.delete("indicator")

        def get_y(v):
            pct = (v - m['min']) / (m['max'] - m['min'])
            return m['h_draw'] - (pct * m['h_draw'])

        bar_x1 = m['bar_x1']
        bar_x2 = m['bar_x2']
        ind_col = self.controller.get_color("span")

        if not m['symmetric']:
            y = get_y(max(m['min'], min(m['max'], value)))
            # Thick pointer
            canvas.create_polygon(bar_x1 - 10, y, bar_x1 - 2, y - 8, bar_x1 - 2, y + 8, fill=ind_col, tags="indicator")
            canvas.create_polygon(bar_x2 + 10, y, bar_x2 + 2, y - 8, bar_x2 + 2, y + 8, fill=ind_col, tags="indicator")
            canvas.create_line(bar_x1, y, bar_x2, y, fill=ind_col, width=4, tags="indicator")
        else:
            if val_b is None: val_b = value
            y_top = get_y(max(m['min'], min(m['max'], val_b)))
            y_bot = get_y(max(m['min'], min(m['max'], value)))
            canvas.create_rectangle(bar_x1 - 4, y_top, bar_x2 + 4, y_bot, fill=ind_col, outline="", tags="indicator")

    def toggle_mode(self):
        self.is_violin_mode = not self.is_violin_mode
        self.btn_mode.config(text="Violin" if self.is_violin_mode else "Chrom")
        self.target_note_key = "Auto"
        self.btn_target.config(text="Tgt: Auto")
        self.violin_target_idx = 0
        with self.pitch_lock:
            self.pitch_history.clear()
            self.audio_buffer.fill(0.0)
            self.prev_pitch_hz = None
            self.jump_counter = 0
        self.rate_display_history.clear()
        self.depth_display_history.clear()
        self.last_display_rate_hz = None
        self.last_display_depth_cents = None
        self.last_vibrato_timestamp = 0.0

    def toggle_analysis(self):
        if not self.is_analyzing:
            if self.controller.audio_device_index is None:
                messagebox.showerror("Error", "No audio device detected")
                return
            self.is_analyzing = True
            self.btn_analyze.config(text="STOP")
            with self.pitch_lock:
                self.pitch_history.clear()
                self.audio_buffer.fill(0.0)
                self.prev_pitch_hz = None
                self.jump_counter = 0
            self.rate_display_history.clear()
            self.depth_display_history.clear()
            self.last_display_rate_hz = None
            self.last_display_depth_cents = None
            self.last_vibrato_timestamp = 0.0
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.analysis_thread.start()
        else:
            self.is_analyzing = False
            self.btn_analyze.config(text="START")
            self.reset_display()

    def _handle_audio_error(self, context, err):
        self.is_analyzing = False
        self.btn_analyze.config(text="START")
        with self.pitch_lock:
            self.prev_pitch_hz = None
        self.reset_display()
        messagebox.showerror("Audio Input Error", f"{context}\n\n{err}")

    def reset_display(self):
        col = self.controller.get_color("fg")
        self.lbl_rate_val.config(text="-- Hz", foreground=col)
        self.lbl_depth_val.config(text="-- ct", foreground=col)
        self.lbl_note.config(text="--", foreground=col)
        self.lbl_freq.config(text="-- Hz")
        self.rate_display_history.clear()
        self.depth_display_history.clear()
        self.last_display_rate_hz = None
        self.last_display_depth_cents = None
        self.last_vibrato_timestamp = 0.0
        self.update_gauge_needle(self.canvas_rate, 0)
        self.update_gauge_needle(self.canvas_depth, 0)

    def analysis_loop(self):
        window_func = np.hanning(self.WINDOW_SIZE)
        try:
            sd.check_input_settings(device=self.controller.audio_device_index, channels=1, samplerate=self.RATE, dtype="float32")
        except Exception as err:
            self.after(0, lambda e=err: self._handle_audio_error("Unable to initialize microphone input.", e))
            return

        try:
            with sd.InputStream(device=self.controller.audio_device_index, channels=1, samplerate=self.RATE, blocksize=self.STEP_SIZE, dtype="float32") as stream:
                while self.is_analyzing:
                    data, _ = stream.read(self.STEP_SIZE)
                    
                    with self.pitch_lock:
                        self.audio_buffer = np.roll(self.audio_buffer, -self.STEP_SIZE)
                        self.audio_buffer[-self.STEP_SIZE:] = data[:, 0]

                        audio_window = self.audio_buffer * window_func
                        prev_f0 = self.prev_pitch_hz
                        
                    audio_window = audio_window - np.mean(audio_window)

                    notes = self.VIOLIN_NOTES if self.is_violin_mode else self.CHROMATIC_NOTES
                    expected_freq = None
                    if self.target_note_key != "Auto" and self.target_note_key in notes:
                        expected_freq = notes[self.target_note_key]
                    elif prev_f0 is not None:
                        expected_freq = prev_f0

                    max_f0 = self.MAX_F0_VIOLIN if self.is_violin_mode else self.MAX_F0_CHROMATIC
                    refined_freq, conf = estimate_fundamental_frequency(
                        audio_window=audio_window,
                        rate=self.RATE,
                        fft_size=self.FFT_SIZE,
                        min_f0=self.MIN_F0,
                        max_f0=max_f0,
                        expected_f0=expected_freq,
                        prev_f0=prev_f0
                    )

                    if refined_freq is None or conf < 0.15:
                        if prev_f0 is not None:
                            refined_freq = prev_f0
                        else:
                            continue  

                    history_copy = None
                    with self.pitch_lock:
                        if len(self.pitch_history) > 0:
                            local_ref = float(np.median(np.array(self.pitch_history)[-8:]))
                            jump_cents = abs(1200.0 * np.log2(max(refined_freq, EPSILON) / max(local_ref, EPSILON)))
                            
                            if jump_cents > self.SEGMENT_RESET_JUMP_CENTS:
                                self.jump_counter += 1
                                if self.jump_counter >= 3:
                                    self.pitch_history.clear()
                                    self.jump_counter = 0
                            else:
                                self.jump_counter = 0

                        self.prev_pitch_hz = refined_freq
                        self.pitch_history.append(refined_freq)
                        
                        if len(self.pitch_history) >= 24:
                            history_copy = np.array(self.pitch_history, dtype=np.float64)
                            
                    if history_copy is not None:
                        ui_data = self._compute_dsp_stats(history_copy)
                        if ui_data and not self.is_processing_stats:
                            self.is_processing_stats = True
                            self.after(0, self._update_vibrato_ui, ui_data)
                        
        except Exception as err:
            self.after(0, lambda e=err: self._handle_audio_error("Microphone stream failed.", e))

    def _compute_dsp_stats(self, raw_freqs):
        freqs = smooth_and_filter_pitch_track(raw_freqs)
        if len(freqs) < 16: return None

        center_freq = float(np.median(freqs))
        if center_freq <= 0: return None

        notes = self.VIOLIN_NOTES if self.is_violin_mode else self.CHROMATIC_NOTES
        if self.target_note_key != "Auto" and self.target_note_key in notes:
            note_str = self.target_note_key
        else:
            note_str = min(notes.keys(), key=lambda n: abs(notes[n] - center_freq))

        cents_deviation = 1200.0 * np.log2(np.maximum(freqs, EPSILON) / center_freq)
        cents_deviation = cents_deviation[np.abs(cents_deviation) <= self.MAX_VALID_CENTS]
        if len(cents_deviation) < 16: return None

        frames_per_sec = self.RATE / float(self.STEP_SIZE)
        trend_frames = max(3, int(frames_per_sec * 0.35))
        trend = _moving_average(cents_deviation, trend_frames)
        vib_signal = cents_deviation - trend
        
        rms = np.sqrt(np.mean(vib_signal**2))
        depth_cents = 2.0 * np.sqrt(2.0) * rms
        depth_cents = float(max(0.0, min(self.MAX_DEPTH_CENTS, depth_cents)))

        rate_hz_raw, mod_confidence = estimate_modulation_rate_hz(
            cents_track=vib_signal, 
            sample_rate_hz=frames_per_sec,
            min_rate_hz=2.0,
            max_rate_hz=12.0
        )

        is_confident_vibrato = (
                depth_cents >= self.VIBRATO_MIN_DEPTH_CENTS and
                mod_confidence >= self.VIBRATO_MIN_CONFIDENCE and
                self.VIBRATO_MIN_RATE_HZ <= rate_hz_raw <= self.VIBRATO_MAX_RATE_HZ
        )

        return {
            "center_freq": center_freq,
            "note_str": note_str,
            "rate_hz_raw": rate_hz_raw,
            "depth_cents_raw": depth_cents,
            "is_confident": is_confident_vibrato
        }

    def _update_vibrato_ui(self, data):
        try:
            if not self.is_analyzing: return

            self.lbl_freq.config(text=f"{data['center_freq']:.1f} Hz")
            self.lbl_note.config(text=data['note_str'])

            now_ts = time.time()
            if data['is_confident']:
                self.last_vibrato_timestamp = now_ts
                self.rate_display_history.append(data['rate_hz_raw'])
                self.depth_display_history.append(data['depth_cents_raw'])

                target_rate_hz = float(np.median(np.array(self.rate_display_history, dtype=np.float64)))
                target_depth_cents = float(np.median(np.array(self.depth_display_history, dtype=np.float64)))

                if self.last_display_rate_hz is None or self.last_display_depth_cents is None:
                    rate_hz = target_rate_hz
                    depth_cents = target_depth_cents
                else:
                    rate_hz = ((1.0 - self.DISPLAY_RATE_ALPHA) * self.last_display_rate_hz) + (self.DISPLAY_RATE_ALPHA * target_rate_hz)
                    depth_cents = ((1.0 - self.DISPLAY_DEPTH_ALPHA) * self.last_display_depth_cents) + (self.DISPLAY_DEPTH_ALPHA * target_depth_cents)

                self.last_display_rate_hz = float(rate_hz)
                self.last_display_depth_cents = float(depth_cents)
            else:
                hold_active = (
                        self.last_display_rate_hz is not None and
                        self.last_display_depth_cents is not None and
                        (now_ts - self.last_vibrato_timestamp) <= self.VIBRATO_HOLD_SECONDS
                )

                if hold_active:
                    rate_hz = float(self.last_display_rate_hz)
                    depth_cents = float(self.last_display_depth_cents)
                else:
                    if self.last_display_rate_hz is None or self.last_display_depth_cents is None:
                        rate_hz = 0.0
                        depth_cents = 0.0
                    else:
                        rate_hz = float(self.last_display_rate_hz) * self.DISPLAY_DECAY
                        depth_cents = float(self.last_display_depth_cents) * self.DISPLAY_DECAY
                        if rate_hz < 0.1: rate_hz = 0.0
                        if depth_cents < 0.5: depth_cents = 0.0

                    self.last_display_rate_hz = float(rate_hz)
                    self.last_display_depth_cents = float(depth_cents)
                    if rate_hz == 0.0 and depth_cents == 0.0:
                        self.rate_display_history.clear()
                        self.depth_display_history.clear()

            self.lbl_rate_val.config(text=f"{rate_hz:.1f} Hz")
            self.lbl_depth_val.config(text=f"{depth_cents:.0f} ct")
            self.update_gauge_needle(self.canvas_rate, rate_hz)
            self.update_gauge_needle(self.canvas_depth, -depth_cents / 2.0, depth_cents / 2.0)

            is_rate_good = (3.0 <= rate_hz <= 9.0)
            is_rate_warn = (2.0 <= rate_hz <= 10.0)

            is_depth_good = (10.0 <= depth_cents <= 40.0)
            is_depth_warn = (depth_cents <= 80.0)

            c_good = self.controller.get_color("good")
            c_warn = self.controller.get_color("warn")
            c_bad = self.controller.get_color("bad")
            c_def = self.controller.get_color("fg")

            color_rate = c_good if is_rate_good else (c_warn if is_rate_warn else c_bad)
            color_depth = c_good if is_depth_good else (c_warn if is_depth_warn else c_bad)

            if rate_hz == 0.0: color_rate = c_def
            if depth_cents == 0.0: color_depth = c_def

            self.lbl_rate_val.config(foreground=color_rate)
            self.lbl_depth_val.config(foreground=color_depth)

            if is_rate_good and is_depth_good and rate_hz > 0:
                self.lbl_note.config(foreground=c_good)
            elif is_rate_warn and is_depth_warn and rate_hz > 0:
                self.lbl_note.config(foreground=c_warn)
            else:
                self.lbl_note.config(foreground=c_def)
                
        finally:
            self.is_processing_stats = False


class SettingsPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="Main.TFrame")
        self.controller = controller

        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(header, text="< Menu", style="Header.TButton", command=lambda: controller.show_frame("StartPage")).pack(side=tk.LEFT)
        
        ttk.Label(self, text="Settings", font=("Helvetica", 18, "bold")).pack(pady=20)

        self.lbl_theme = ttk.Label(self, text=f"Theme: {controller.current_theme}", font=("Helvetica", 14))
        self.lbl_theme.pack(pady=10)

        ttk.Button(self, text="Cycle Color Theme", style="Action.TButton", command=self.change_theme).pack(pady=20)

    def change_theme(self):
        new_theme = self.controller.cycle_theme()
        self.lbl_theme.config(text=f"Theme: {new_theme}")


class TunerPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="Main.TFrame")
        self.controller = controller
        self.is_listening = False
        self.stream_thread = None
        self.is_violin_mode = True

        # UPDATED: I2S mics prefer 48000Hz on RPi
        self.RATE = 48000
        self.CHUNK = 1024
        self.WINDOW_SIZE = 2048
        self.STEP_SIZE = 1024
        self.FFT_SIZE = 8192
        self.MIN_FREQ = 180.0
        self.MAX_F0_VIOLIN = 760.0
        self.MAX_F0_NATURAL = 1200.0
        self.VIOLIN_TARGETS = {"G3": 196.00, "D4": 293.66, "A4": 440.00, "E5": 659.25}

        self.NATURAL_TARGETS = {}
        concert_a = 440.0
        for midi in range(55, 97):
            note_idx = midi % 12
            if note_idx in [0, 2, 4, 5, 7, 9, 11]:
                freq = concert_a * (2 ** ((midi - 69) / 12.0))
                name = {0: 'C', 2: 'D', 4: 'E', 5: 'F', 7: 'G', 9: 'A', 11: 'B'}[note_idx]
                octave = (midi // 12) - 1
                self.NATURAL_TARGETS[f"{name}{octave}"] = freq

        self.pitch_lock = threading.Lock()
        self.is_processing_ui = False

        self.pitch_history = deque(maxlen=32)
        self.audio_buffer = np.zeros(self.WINDOW_SIZE)
        self.prev_pitch_hz = None
        self.jump_counter = 0

        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=2, padx=2)
        ttk.Button(header, text="< Menu", style="Header.TButton", command=self.go_back).pack(side=tk.LEFT, padx=2)
        self.mode_btn = ttk.Button(header, text="Mode: Vln", style="Header.TButton", command=self.toggle_mode)
        self.mode_btn.pack(side=tk.RIGHT, padx=2)

        display_frame = ttk.Frame(self, style="Main.TFrame")
        display_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        self.note_var = tk.StringVar(value="--")
        self.note_label = ttk.Label(display_frame, textvariable=self.note_var, font=("Helvetica", 64, "bold"), anchor="center")
        self.note_label.pack(pady=(10, 0))

        self.freq_var = tk.StringVar(value="0.0 Hz")
        self.freq_label = ttk.Label(display_frame, textvariable=self.freq_var, font=("Helvetica", 14))
        self.freq_label.pack()

        # Thicker horizontal gauge
        self.gauge_canvas = tk.Canvas(display_frame, height=50, highlightthickness=0)
        self.gauge_canvas.pack(fill=tk.X, pady=15)
        self.gauge_canvas.bind("<Configure>", lambda e: self.draw_gauge_background())

        self.start_button = ttk.Button(display_frame, text="START", style="Action.TButton", command=self.toggle_listening)
        self.start_button.pack(pady=5)

    def on_show(self):
        self.gauge_canvas.configure(bg=self.controller.THEMES[self.controller.current_theme]["bg"])
        self.draw_gauge_background()

    def go_back(self):
        self.stop_listening()
        self.controller.show_frame("StartPage")

    def toggle_mode(self):
        self.is_violin_mode = not self.is_violin_mode
        self.mode_btn.config(text="Mode: Nat" if not self.is_violin_mode else "Mode: Vln")
        with self.pitch_lock:
            self.pitch_history.clear()
            self.audio_buffer.fill(0.0)
            self.prev_pitch_hz = None
            self.jump_counter = 0

    def draw_gauge_background(self):
        self.gauge_canvas.delete("all")
        cw = self.gauge_canvas.winfo_width()
        if cw < 50: cw = 300
        ch = 50
        
        c_good_bg = self.controller.get_color("good_bg")
        c_warn_bg = self.controller.get_color("warn_bg")
        c_bad_bg = self.controller.get_color("bad_bg")

        # Flat Horizontal Blocks
        w_bad = cw * 0.25
        w_warn = cw * 0.15
        w_good = cw * 0.20
        
        x = 0
        self.gauge_canvas.create_rectangle(x, 10, x+w_bad, ch-10, fill=c_bad_bg, outline="")
        x += w_bad
        self.gauge_canvas.create_rectangle(x, 10, x+w_warn, ch-10, fill=c_warn_bg, outline="")
        x += w_warn
        self.gauge_canvas.create_rectangle(x, 10, x+w_good, ch-10, fill=c_good_bg, outline="")
        x += w_good
        self.gauge_canvas.create_rectangle(x, 10, x+w_warn, ch-10, fill=c_warn_bg, outline="")
        x += w_warn
        self.gauge_canvas.create_rectangle(x, 10, cw, ch-10, fill=c_bad_bg, outline="")
        
        self.gauge_canvas.create_line(cw/2, 5, cw/2, ch-5, fill=self.controller.get_color("fg"), width=3)

    def update_gauge(self, cents_off):
        cw = self.gauge_canvas.winfo_width()
        if cw < 50: cw = 300
        self.gauge_canvas.delete("indicator")
        
        center_x = cw / 2
        offset_px = max(-center_x + 10, min(center_x - 10, cents_off * (cw / 100)))
        new_x = center_x + offset_px
        
        # Draw huge pointer block
        ind_col = self.controller.get_color("span")
        self.gauge_canvas.create_polygon(new_x - 10, 5, new_x + 10, 5, new_x, 15, fill=ind_col, tags="indicator")
        self.gauge_canvas.create_polygon(new_x - 10, 45, new_x + 10, 45, new_x, 35, fill=ind_col, tags="indicator")
        self.gauge_canvas.create_line(new_x, 5, new_x, 45, fill=ind_col, width=4, tags="indicator")

        abs_cents = abs(cents_off)
        if abs_cents < 5:
            new_color = self.controller.get_color("good")
        elif abs_cents < 25:
            new_color = self.controller.get_color("warn")
        else:
            new_color = self.controller.get_color("bad")
        self.note_label.config(foreground=new_color)

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if not self.is_listening:
            if self.controller.audio_device_index is None:
                messagebox.showerror("Error", "No audio device detected")
                return
            self.is_listening = True
            self.start_button.config(text="STOP")
            with self.pitch_lock:
                self.pitch_history.clear()
                self.audio_buffer.fill(0.0)
                self.prev_pitch_hz = None
                self.jump_counter = 0
            self.stream_thread = threading.Thread(target=self.audio_loop, daemon=True)
            self.stream_thread.start()

    def stop_listening(self):
        self.is_listening = False
        self.start_button.config(text="START")
        col = self.controller.get_color("fg")
        self.note_var.set("--")
        self.note_label.config(foreground=col)
        self.freq_var.set("0.0 Hz")
        with self.pitch_lock:
            self.prev_pitch_hz = None
        self.update_gauge(0)

    def _handle_audio_error(self, context, err):
        self.is_listening = False
        self.start_button.config(text="START")
        self.note_var.set("--")
        self.freq_var.set("0.0 Hz")
        with self.pitch_lock:
            self.prev_pitch_hz = None
        self.update_gauge(0)
        messagebox.showerror("Audio Input Error", f"{context}\n\n{err}")

    def audio_loop(self):
        window_func = np.hanning(self.WINDOW_SIZE)
        try:
            sd.check_input_settings(device=self.controller.audio_device_index, channels=1, samplerate=self.RATE, dtype="float32")
        except Exception as err:
            self.after(0, lambda e=err: self._handle_audio_error("Unable to initialize microphone input.", e))
            return

        try:
            with sd.InputStream(device=self.controller.audio_device_index, channels=1, samplerate=self.RATE, blocksize=self.CHUNK, dtype="float32") as stream:
                while self.is_listening:
                    data, _ = stream.read(self.CHUNK)

                    with self.pitch_lock:
                        self.audio_buffer = np.roll(self.audio_buffer, -self.STEP_SIZE)
                        self.audio_buffer[-self.STEP_SIZE:] = data[-self.STEP_SIZE:, 0]
                        
                        audio_window = self.audio_buffer * window_func
                        prev_f0 = self.prev_pitch_hz

                    audio_window = audio_window - np.mean(audio_window)

                    max_f0 = self.MAX_F0_VIOLIN if self.is_violin_mode else self.MAX_F0_NATURAL
                    refined_freq, _ = estimate_fundamental_frequency(
                        audio_window=audio_window,
                        rate=self.RATE,
                        fft_size=self.FFT_SIZE,
                        min_f0=self.MIN_FREQ,
                        max_f0=max_f0,
                        expected_f0=prev_f0,
                        prev_f0=prev_f0
                    )
                    
                    if refined_freq is None:
                        continue

                    history_copy = None
                    with self.pitch_lock:
                        if len(self.pitch_history) > 0:
                            local_ref = float(np.median(np.array(self.pitch_history)[-8:]))
                            jump_cents = abs(1200.0 * np.log2(max(refined_freq, EPSILON) / max(local_ref, EPSILON)))
                            if jump_cents > 220.0:
                                self.jump_counter += 1
                                if self.jump_counter >= 3:
                                    self.pitch_history.clear()
                                    self.jump_counter = 0
                            else:
                                self.jump_counter = 0

                        self.prev_pitch_hz = refined_freq
                        self.pitch_history.append(refined_freq)
                        if len(self.pitch_history) >= 7:
                            history_copy = np.array(self.pitch_history, dtype=np.float64)

                    if history_copy is not None:
                        stable_track = smooth_and_filter_pitch_track(history_copy)
                        if len(stable_track) > 0:
                            stable_freq = float(np.median(stable_track))
                        else:
                            stable_freq = float(refined_freq)

                        targets = self.VIOLIN_TARGETS if self.is_violin_mode else self.NATURAL_TARGETS
                        closest = min(targets, key=lambda k: abs(targets[k] - stable_freq))
                        target_f = targets[closest]
                        cents = 1200.0 * np.log2(max(stable_freq, EPSILON) / max(target_f, EPSILON))

                        if not self.is_processing_ui:
                            self.is_processing_ui = True
                            self.after(0, lambda n=closest, f=stable_freq, c=cents: self.update_tuner_ui(n, f, c))
                            
        except Exception as err:
            self.after(0, lambda e=err: self._handle_audio_error("Microphone stream failed.", e))

    def update_tuner_ui(self, note, freq, cents):
        try:
            self.note_var.set(note)
            self.freq_var.set(f"{freq:.1f} Hz")
            self.update_gauge(cents)
        finally:
            self.is_processing_ui = False


if __name__ == "__main__":
    try:
        app = VibratoApp()
        app.protocol("WM_DELETE_WINDOW", app.quit)
        app.mainloop()
    except Exception as e:
        print(f"Error: {e}")
