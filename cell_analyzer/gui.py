"""
cell_analyzer/gui.py -- graphical front-end of Cell Analyzer

    python -m cell_analyzer          (from the cell-analyzer folder)

* Every parameter from config.PARAM_SCHEMA gets a widget automatically.
* Choosing a folder opens a dialog that asks which file format to analyse and
  which file-name / scene-name filters to apply (e.g. "_merged", "stitched").
* Settings can be saved/loaded as JSON; a previous analysis_log_*.json can be
  loaded too (its parameters are restored).
* The pipeline runs in a separate process (the GUI stays responsive, output
  is streamed into the Run-log tab, "Stop" cancels cleanly).
"""

import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, simpledialog

from . import config as pc

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent          # the cell-analyzer folder

LAST_SETTINGS = Path.home() / ".cell_analyzer_last_settings.json"
OLD_LAST_SETTINGS = Path.home() / ".cell_pipeline_gui_last.json"  # before the rename
HELP_COLOR = "#6b6b6b"

# which schema sections go on which tab
TABS = [
    ("Input / Output", ["Input", "Output"]),
    ("Stages & Channels", ["General", "Channels", "Physical sizes"]),
    ("Background", ["Background"]),
    ("Segmentation", ["Segmentation"]),
    ("Non-confluent", ["Non-confluent"]),
    ("Measurement", ["Measurement"]),
    ("Aggregation", ["Aggregation"]),
    ("Analysis plots", ["Analysis"]),
]


# -------------------------------------------------
# Small widgets
# -------------------------------------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        if text:
            widget.bind("<Enter>", self.show, add="+")
            widget.bind("<Leave>", self.hide, add="+")

    def show(self, _=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid",
                 borderwidth=1, wraplength=420, padx=6, pady=4).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.canvas.bind("<Enter>", lambda e: self._bind(True))
        self.canvas.bind("<Leave>", lambda e: self._bind(False))

    def _bind(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        else:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    def _wheel(self, e):
        step = -e.delta if sys.platform == "darwin" else -int(e.delta / 120)
        self.canvas.yview_scroll(step, "units")


# -------------------------------------------------
# Folder set-up dialog: format + name filters
# -------------------------------------------------
class FolderSetupDialog(tk.Toplevel):
    def __init__(self, gui, folder):
        super().__init__(gui)
        self.gui, self.folder = gui, Path(folder)
        self.title("Folder input -- format and name filters")
        self.transient(gui)
        self.result = False
        v = gui.vars
        self.fmt = tk.StringVar(value=v["file_format"].get())
        self.recursive = tk.BooleanVar(value=v["recursive"].get())
        self.f_inc = tk.StringVar(value=v["filename_include"].get())
        self.f_exc = tk.StringVar(value=v["filename_exclude"].get())
        self.s_inc = tk.StringVar(value=v["scene_include"].get())
        self.s_exc = tk.StringVar(value=v["scene_exclude"].get())
        self.case = tk.BooleanVar(value=v["filter_case_sensitive"].get())
        self.regex = tk.BooleanVar(value=v["filter_use_regex"].get())

        pad = dict(padx=8, pady=3)
        ttk.Label(self, text=f"Folder:  {self.folder}", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)

        self.fmt_frame = ttk.LabelFrame(self, text="1.  Which file format are you looking for?")
        self.fmt_frame.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Checkbutton(self, text="Search sub-folders too", variable=self.recursive,
                        command=self._rescan).grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        flt = ttk.LabelFrame(self, text="2.  Should a name filter be applied?  (comma-separated, ANY match)")
        flt.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        rows = [("File name must contain", self.f_inc, "e.g.  _merged, stitched   (blank = all files)"),
                ("File name must NOT contain", self.f_exc, "e.g.  _raw, test"),
                ("Scene name must contain", self.s_inc, "multi-scene files such as .lif, e.g. _merged"),
                ("Scene name must NOT contain", self.s_exc, "")]
        for i, (lab, var, hint) in enumerate(rows):
            ttk.Label(flt, text=lab).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            ttk.Entry(flt, textvariable=var, width=34).grid(row=i, column=1, sticky="w", padx=6, pady=2)
            ttk.Label(flt, text=hint, foreground=HELP_COLOR).grid(row=i, column=2, sticky="w", padx=6)
            var.trace_add("write", lambda *a: self._schedule_update())
        ttk.Checkbutton(flt, text="case-sensitive", variable=self.case,
                        command=self._schedule_update).grid(row=4, column=0, sticky="w", padx=6)
        ttk.Checkbutton(flt, text="regular expressions", variable=self.regex,
                        command=self._schedule_update).grid(row=4, column=1, sticky="w", padx=6)

        self.count_lbl = ttk.Label(self, text="")
        self.count_lbl.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        lf = ttk.Frame(self)
        lf.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        self.listbox = tk.Listbox(lf, width=100, height=12)
        sb = ttk.Scrollbar(lf, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        ttk.Label(self, text="Scene filters are applied when the files are opened (use 'Preview inputs' "
                             "on the main window to see the scenes).", foreground=HELP_COLOR).grid(
            row=6, column=0, columnspan=3, sticky="w", **pad)

        bf = ttk.Frame(self)
        bf.grid(row=7, column=0, columnspan=3, sticky="e", **pad)
        ttk.Button(bf, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(bf, text="OK", command=self._ok).pack(side="right", padx=4)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        self._after = None
        self._rescan()
        self.grab_set()

    def _rescan(self):
        from .io_utils import detect_formats_in_folder
        for w in self.fmt_frame.winfo_children():
            w.destroy()
        try:
            counts = detect_formats_in_folder(self.folder, self.recursive.get())
        except Exception as e:
            counts = {}
            messagebox.showerror("Folder", str(e), parent=self)
        if counts and counts.get(self.fmt.get(), 0) == 0:
            self.fmt.set(max(counts, key=counts.get))
        for i, fmt in enumerate(pc.SUPPORTED_FORMATS):
            n = counts.get(fmt, 0)
            rb = ttk.Radiobutton(self.fmt_frame, text=f".{fmt}   ({n} file{'s' if n != 1 else ''})",
                                 value=fmt, variable=self.fmt, command=self._schedule_update)
            rb.grid(row=0, column=i, padx=8, pady=4, sticky="w")
            if n == 0:
                rb.state(["disabled"])
        if not counts:
            ttk.Label(self.fmt_frame, text="No supported image files found in this folder.",
                      foreground="#b00020").grid(row=1, column=0, columnspan=5, sticky="w", padx=8)
        self._update()

    def _schedule_update(self):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(250, self._update)

    def _update(self):
        from .io_utils import find_input_files
        cfg = pc.default_config()
        cfg.update(input_path=str(self.folder), input_mode="folder", file_format=self.fmt.get(),
                   recursive=self.recursive.get(), filename_include=self.f_inc.get(),
                   filename_exclude=self.f_exc.get(), filter_case_sensitive=self.case.get(),
                   filter_use_regex=self.regex.get())
        self.listbox.delete(0, "end")
        try:
            files = find_input_files(cfg)
        except Exception as e:
            self.count_lbl.configure(text=f"⚠ {e}", foreground="#b00020")
            return
        self.count_lbl.configure(text=f"{len(files)} .{self.fmt.get()} file(s) match:", foreground="")
        for f in files[:500]:
            self.listbox.insert("end", str(f.relative_to(self.folder)))
        if len(files) > 500:
            self.listbox.insert("end", f"... and {len(files) - 500} more")

    def _ok(self):
        v = self.gui.vars
        v["input_path"].set(str(self.folder))
        v["input_mode"].set("folder")
        v["file_format"].set(self.fmt.get())
        v["recursive"].set(self.recursive.get())
        v["filename_include"].set(self.f_inc.get())
        v["filename_exclude"].set(self.f_exc.get())
        v["scene_include"].set(self.s_inc.get())
        v["scene_exclude"].set(self.s_exc.get())
        v["filter_case_sensitive"].set(self.case.get())
        v["filter_use_regex"].set(self.regex.get())
        self.result = True
        self.destroy()


def explain_exit_code(code):
    """Plain-language hint for the crash codes Windows/macOS report instead of a Python error."""
    if code is None:
        return ""
    if code < 0:
        sig = -code
        names = {2: "it was stopped (Stop button / Ctrl+C)", 9: "it was killed, often because the computer ran out of memory",
                 11: "it crashed inside a compiled library (segmentation fault)", 15: "it was terminated"}
        return f"The analysis process ended by signal {sig}: {names.get(sig, 'it was stopped by the system')}."
    u = code & 0xFFFFFFFF
    win = {
        0xC06D007F: "Windows could not find a function in a DLL (0xC06D007F). Usually another program's DLLs are "
                    "loaded instead of the environment's own. Start the program with start_windows.bat (v2.6 or "
                    "newer) -- if it persists, run install_windows.bat --fresh.",
        0xC06D007E: "Windows could not find a DLL (0xC06D007E). Run install_windows.bat again (or with --fresh).",
        0xC0000005: "The process crashed inside a compiled library (access violation, 0xC0000005). Often a damaged "
                    "environment -- run the installer with --fresh -- or a corrupt image file.",
        0xC0000409: "The process crashed inside a compiled library (0xC0000409). Run the installer with --fresh.",
        0xC0000135: "A required DLL is missing (0xC0000135). Run install_windows.bat again.",
        0xC00000FD: "Stack overflow (0xC00000FD).",
        0xC000013A: "The process was stopped (Ctrl+C / window closed).",
        0xC0000017: "Out of memory (0xC0000017). Lower 'Parallel worker processes'.",
    }
    if u in win:
        return win[u]
    if u > 255:
        return (f"Exit code {code} = 0x{u:08X}: the process crashed outside Python, so there is no error message. "
                f"Lower 'Parallel worker processes' and try again; if it persists, reinstall with --fresh.")
    return ""


# -------------------------------------------------
# Main window
# -------------------------------------------------
class CellAnalyzerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{pc.APP_NAME}  (v{pc.PIPELINE_VERSION})")
        self.geometry("1180x820")
        self.minsize(900, 600)
        self.vars, self.field_widgets, self.help_labels = {}, {}, {}
        self.proc, self.q = None, queue.Queue()

        style = ttk.Style(self)
        if "clam" in style.theme_names() and sys.platform.startswith("linux"):
            style.theme_use("clam")
        style.configure("Section.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"))
        style.configure("Horizontal.TProgressbar", background="#2a78d6")  # visible fill on non-native themes

        self._build_vars()
        self._build_ui()
        self._load_last()
        self._refresh_enabled()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # ---------- variables ----------
    def _build_vars(self):
        for key, f in pc.ALL_FIELDS.items():
            if f["type"] == "bool":
                var = tk.BooleanVar(value=f["default"])
            else:
                var = tk.StringVar(value="" if f["default"] is None else str(f["default"]))
            var.trace_add("write", lambda *a: self._refresh_enabled())
            self.vars[key] = var

    def get_config(self):
        """Collect + type-check all widget values. Raises ValueError with a readable message."""
        cfg, errs = {}, []
        for key, f in pc.ALL_FIELDS.items():
            try:
                cfg[key] = pc.coerce(key, self.vars[key].get())
            except Exception as e:
                errs.append(f"• {f['label']}: {e}")
        if errs:
            raise ValueError("Please fix these values:\n\n" + "\n".join(errs))
        return cfg

    def set_config(self, cfg):
        for key, val in cfg.items():
            if key in self.vars:
                self.vars[key].set("" if val is None else val)
        self._refresh_enabled()

    # ---------- UI ----------
    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        sections = dict(pc.PARAM_SCHEMA)

        for tab_name, section_names in TABS:
            sf = ScrollFrame(self.nb)
            self.nb.add(sf, text=tab_name)
            body = sf.inner
            body.columnconfigure(1, weight=0)
            body.columnconfigure(2, weight=1)
            row = 0
            sf.canvas.bind("<Configure>", lambda e, b=body, c=sf.canvas: self._rewrap(b, c, e.width), add="+")
            if tab_name == "Input / Output":
                row = self._build_input_header(body, row)
            if tab_name == "Stages & Channels":
                ttk.Button(body, text="Read channel info from first input file…",
                           command=self._inspect_first_file).grid(row=row, column=0, columnspan=2,
                                                                  sticky="w", padx=10, pady=(10, 0))
                row += 1
            if tab_name == "Non-confluent":
                ttk.Label(body, wraplength=900, foreground=HELP_COLOR, justify="left", text=(
                    "Scenes with fewer nuclei than the threshold are 'non-confluent'. If non-confluent "
                    "analysis is enabled, the cell watershed in those scenes is restricted to a foreground "
                    "mask so cells don't flood across empty space. If it is disabled, you choose whether such "
                    "scenes are segmented like confluent ones or skipped entirely. Every scene's decision is "
                    "stored in <dataset>/segmentation/<dataset>_segmentation_summary.csv and in the "
                    "'scene_confluent' column of the measurements.")).grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 0))
                row += 1
            for sec in section_names:
                ttk.Label(body, text=sec, style="Section.TLabel").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(14, 4))
                row += 1
                for f in sections[sec]:
                    if f["key"] == "input_path":
                        continue  # custom header widget
                    row = self._build_field(body, f, row)

        # run-log tab
        logf = ttk.Frame(self.nb)
        self.nb.add(logf, text="Run log")
        self.log = tk.Text(logf, wrap="none", font=("Menlo" if sys.platform == "darwin" else "Courier", 10))
        ys = ttk.Scrollbar(logf, command=self.log.yview)
        xs = ttk.Scrollbar(logf, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys.set, xscrollcommand=xs.set, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        logf.rowconfigure(0, weight=1)
        logf.columnconfigure(0, weight=1)
        self.log.tag_configure("err", foreground="#b00020")
        self.log.tag_configure("ok", foreground="#1b7f3b")
        self.log.tag_configure("head", font=("TkDefaultFont", 10, "bold"))

        # bottom bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Button(bar, text="Load settings…", command=self._load_settings).pack(side="left", padx=2)
        ttk.Button(bar, text="Save settings…", command=self._save_settings).pack(side="left", padx=2)
        ttk.Button(bar, text="Reset to defaults", command=self._reset).pack(side="left", padx=2)
        ttk.Button(bar, text="Check settings", command=self._check).pack(side="left", padx=2)
        ttk.Button(bar, text="Open output folder", command=self._open_output).pack(side="left", padx=2)
        ttk.Button(bar, text="Open plots", command=self._open_plots).pack(side="left", padx=2)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=2)
        self.run_btn = ttk.Button(bar, text="▶  Run analysis", style="Run.TButton", command=self._run)
        self.run_btn.pack(side="right", padx=2)
        self.progress = ttk.Progressbar(bar, length=220, mode="determinate")
        self.progress.pack(side="right", padx=8)
        self.status = ttk.Label(bar, text="Ready", width=34, anchor="e")
        self.status.pack(side="right")

    def _build_input_header(self, body, row):
        box = ttk.LabelFrame(body, text="What should be analysed?")
        box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 0))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Input file or folder").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(box, textvariable=self.vars["input_path"]).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Single file…", command=self._pick_file).grid(row=0, column=2, padx=2)
        ttk.Button(box, text="Folder…", command=self._pick_folder).grid(row=0, column=3, padx=2)
        ttk.Button(box, text="Formats & filters…", command=self._folder_dialog).grid(row=0, column=4, padx=2)
        ttk.Button(box, text="Preview inputs", command=self._preview).grid(row=0, column=5, padx=(2, 6))
        self.input_info = ttk.Label(box, text="", foreground=HELP_COLOR)
        self.input_info.grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 4))
        return row + 1

    def _build_field(self, body, f, row):
        key, t = f["key"], f["type"]
        var = self.vars[key]
        lbl = ttk.Label(body, text=f["label"] + ("  (advanced)" if f["advanced"] else ""))
        lbl.grid(row=row, column=0, sticky="w", padx=(20, 8), pady=3)
        cell = ttk.Frame(body)
        cell.grid(row=row, column=1, sticky="w", pady=3)
        widgets = [lbl]

        if t == "bool":
            w = ttk.Checkbutton(cell, variable=var)
            w.pack(side="left")
            widgets.append(w)
        elif t == "choice":
            w = ttk.Combobox(cell, textvariable=var, values=f["choices"], state="readonly", width=18)
            w.pack(side="left")
            widgets.append(w)
        elif t in ("dir", "file_open", "file_save", "channel_paths", "path_in"):
            w = ttk.Entry(cell, textvariable=var, width=44)
            w.pack(side="left")
            widgets.append(w)
            if t == "dir":
                cmd = lambda v=var: self._browse_dir(v)
            elif t == "file_open":
                cmd = lambda v=var: self._browse_open(v, [("Excel", "*.xlsx *.xls"), ("All files", "*")])
            elif t == "file_save":
                cmd = lambda v=var: self._browse_save(v)
            elif t == "channel_paths":
                cmd = lambda v=var: self._add_background_file(v)
            else:
                cmd = lambda v=var: self._pick_file()
            b = ttk.Button(cell, text="Add…" if t == "channel_paths" else "Browse…", command=cmd)
            b.pack(side="left", padx=4)
            widgets.append(b)
        else:
            width = 36 if t in ("str", "str_list") else 12
            w = ttk.Entry(cell, textvariable=var, width=width)
            w.pack(side="left")
            widgets.append(w)
            if t == "float_opt":
                ttk.Label(cell, text="(blank = auto)", foreground=HELP_COLOR).pack(side="left", padx=4)

        help_lbl = ttk.Label(body, text=f["help"], foreground=HELP_COLOR, wraplength=340, justify="left")
        help_lbl.grid(row=row, column=2, sticky="w", padx=8)
        widgets.append(help_lbl)
        self.help_labels.setdefault(str(body), []).append(help_lbl)
        ToolTip(lbl, f["help"])
        self.field_widgets[key] = widgets
        return row + 1

    def _rewrap(self, body, canvas, width):
        canvas.itemconfigure(canvas.find_all()[0], width=width)
        body.update_idletasks()
        for lbl in self.help_labels.get(str(body), []):
            lbl.configure(wraplength=max(180, width - lbl.winfo_x() - 16))

    def _refresh_enabled(self):
        if not self.field_widgets:
            return
        for key, f in pc.ALL_FIELDS.items():
            cond = f.get("enabled_if")
            if not cond or key not in self.field_widgets:
                continue
            src, allowed = cond
            val = self.vars[src].get()
            on = val in allowed
            for w in self.field_widgets[key]:
                try:
                    if isinstance(w, ttk.Combobox):
                        w.configure(state="readonly" if on else "disabled")
                    else:
                        w.state(["!disabled"] if on else ["disabled"])
                except Exception:
                    pass
        self._update_input_info()

    def _update_input_info(self):
        if not hasattr(self, "input_info"):
            return
        p = self.vars["input_path"].get()
        if not p:
            txt = "Choose a single file (.lif / .czi / ...) or a folder."
        elif Path(p).is_dir():
            inc = self.vars["filename_include"].get() or "—"
            sinc = self.vars["scene_include"].get() or "—"
            txt = (f"Folder mode · format .{self.vars['file_format'].get()} · file name contains: {inc} · "
                   f"scene name contains: {sinc}")
        elif Path(p).is_file():
            from .io_utils import format_of
            txt = f"Single-file mode · format: {format_of(p) or 'unknown'} · scene name contains: " \
                  f"{self.vars['scene_include'].get() or '—'}"
        else:
            txt = "⚠ path does not exist"
        self.input_info.configure(text=txt)

    # ---------- browse helpers ----------
    def _browse_dir(self, var):
        d = filedialog.askdirectory(initialdir=var.get() or None)
        if d:
            var.set(d)

    def _browse_open(self, var, types):
        f = filedialog.askopenfilename(filetypes=types)
        if f:
            var.set(f)

    def _browse_save(self, var):
        f = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv"), ("Parquet", "*.parquet")])
        if f:
            var.set(f)

    def _add_background_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Background field", "*.npy"), ("All files", "*")])
        if not f:
            return
        ch = simpledialog.askinteger("Channel", f"Which channel index is\n{Path(f).name}\nfor?",
                                     parent=self, minvalue=0)
        if ch is None:
            return
        cur = pc.parse_channel_paths(var.get()) if var.get().strip() else {}
        cur[ch] = f
        var.set("; ".join(f"{k}={v}" for k, v in sorted(cur.items())))

    def _pick_file(self):
        # macOS aborts the whole program (NSException) on file types with two dots such as "*.ome.tif",
        # so only the last part of each extension is offered (".tif" already covers ".ome.tif").
        exts = sorted({"*." + e.rsplit(".", 1)[-1] for es in pc.SUPPORTED_FORMATS.values() for e in es})
        f = filedialog.askopenfilename(filetypes=[("Images", " ".join(exts)), ("All files", "*")])
        if f:
            from .io_utils import format_of
            self.vars["input_path"].set(f)
            self.vars["input_mode"].set("file")
            fmt = format_of(f)
            if fmt:
                self.vars["file_format"].set(fmt)
            if not self.vars["output_root"].get():
                self.vars["output_root"].set(str(Path(f).parent / "analysis_output"))

    def _pick_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        dlg = FolderSetupDialog(self, d)
        self.wait_window(dlg)
        if dlg.result and not self.vars["output_root"].get():
            self.vars["output_root"].set(str(Path(d) / "analysis_output"))

    def _folder_dialog(self):
        p = self.vars["input_path"].get()
        if p and Path(p).is_dir():
            self.wait_window(FolderSetupDialog(self, p))
        elif p and Path(p).is_file():
            messagebox.showinfo("Single file", "A single file is selected -- only the scene filter "
                                "(Input section below) applies.")
        else:
            self._pick_folder()

    # ---------- preview / inspect (threaded, may open many files) ----------
    def _preview(self):
        try:
            cfg = self.get_config()
        except ValueError as e:
            return messagebox.showerror("Settings", str(e))
        if not cfg["input_path"] or not Path(cfg["input_path"]).exists():
            return messagebox.showerror("Input", "Choose an existing input file or folder first.")
        self.status.configure(text="Scanning inputs…")
        win = tk.Toplevel(self)
        win.title("Input preview")
        win.geometry("900x520")
        lbl = ttk.Label(win, text="Opening files and reading scene names…")
        lbl.pack(anchor="w", padx=8, pady=6)
        tree = ttk.Treeview(win, columns=("scenes",), show="tree headings")
        tree.heading("#0", text="File / scene")
        tree.heading("scenes", text="Scenes used / total")
        tree.column("#0", width=680)
        tree.pack(fill="both", expand=True, padx=8, pady=4)

        def work():
            from . import io_utils
            msgs = []
            try:
                ds = io_utils.resolve_jobs(cfg, log=msgs.append)
                err = None
            except Exception as e:
                ds, err = [], str(e)
            self.after(0, lambda: fill(ds, msgs, err))

        def fill(ds, msgs, err):
            self.status.configure(text="Ready")
            if not win.winfo_exists():
                return
            if err:
                lbl.configure(text=f"⚠ {err}", foreground="#b00020")
                return
            n_sc = sum(len(d["scenes"]) for d in ds)
            lbl.configure(text=f"{len(ds)} file(s), {n_sc} scene(s) will be analysed."
                               + (f"   ⚠ {len(msgs)} warning(s)" if msgs else ""))
            for d in ds:
                node = tree.insert("", "end", text=f"{d['file'].name}   →  output folder: {d['stem']}",
                                   values=(f"{len(d['scenes'])} / {d['n_scenes_total']}",), open=len(ds) == 1)
                for s in d["scenes"][:2000]:
                    tree.insert(node, "end", text=s)
            for m in msgs:
                tree.insert("", "end", text=m)

        threading.Thread(target=work, daemon=True).start()

    def _inspect_first_file(self):
        try:
            cfg = self.get_config()
            from . import io_utils
            files = io_utils.find_input_files(cfg)
        except Exception as e:
            return messagebox.showerror("Inspect", str(e))
        if not files:
            return messagebox.showerror("Inspect", "No input files match the current settings.")

        def work():
            from . import io_utils
            try:
                info = io_utils.inspect_file(files[0])
                self.after(0, lambda: show(info))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Inspect", f"Could not read {files[0].name}:\n{e}"))

        def show(info):
            self.status.configure(text="Ready")
            txt = (f"File: {Path(info['file']).name}\nScenes: {info['n_scenes']}\nDims: {info['dims']}\n"
                   f"Pixel size: {info['pixel_size_um']} µm\nChannels: {', '.join(info['channel_names'])}\n\n"
                   "Use these channel names for the measurement columns?")
            if info["channel_names"] and messagebox.askyesno("Channel info", txt):
                self.vars["channel_names"].set(", ".join(info["channel_names"]))
            elif not info["channel_names"]:
                messagebox.showinfo("Channel info", txt.rsplit("\n\n", 1)[0])

        self.status.configure(text=f"Reading {files[0].name}…")
        threading.Thread(target=work, daemon=True).start()

    # ---------- settings ----------
    def _load_settings(self):
        f = filedialog.askopenfilename(title="Load settings or a previous analysis log",
                                       filetypes=[("JSON", "*.json"), ("All files", "*")])
        if f:
            try:
                self.set_config(pc.load_config_file(f))
                self.status.configure(text=f"Loaded {Path(f).name}")
            except Exception as e:
                messagebox.showerror("Load settings", str(e))

    def _save_settings(self):
        try:
            cfg = self.get_config()
        except ValueError as e:
            return messagebox.showerror("Settings", str(e))
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")],
                                         initialfile="cell_analyzer_settings.json")
        if f:
            pc.save_config_file(cfg, f)
            self.status.configure(text=f"Saved {Path(f).name}")

    def _reset(self):
        if messagebox.askyesno("Reset", "Reset ALL parameters to their defaults?\n"
                                        "(input and output folders are kept)"):
            keep = {k: self.vars[k].get() for k in ("input_path", "output_root")}
            self.set_config({**pc.default_config(), **keep})

    def _load_last(self):
        try:
            for f in (LAST_SETTINGS, OLD_LAST_SETTINGS):
                if f.exists():
                    self.set_config(pc.load_config_file(f))
                    break
        except Exception:
            pass

    def _save_last(self):
        try:
            pc.save_config_file(self.get_config(), LAST_SETTINGS)
        except Exception:
            pass

    def _check(self, quiet_ok=False):
        try:
            cfg = self.get_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return None
        problems = pc.validate_config(cfg)
        if problems:
            messagebox.showerror("Settings", "Please fix:\n\n" + "\n".join(f"• {p}" for p in problems))
            return None
        if not quiet_ok:
            messagebox.showinfo("Settings", "All settings look valid.")
        return cfg

    def _open_output(self):
        p = self.vars["output_root"].get()
        if not p or not Path(p).exists():
            return messagebox.showinfo("Output", "Output folder does not exist yet.")
        if sys.platform == "darwin":
            subprocess.Popen(["open", p])
        elif os.name == "nt":
            os.startfile(p)  # noqa
        else:
            subprocess.Popen(["xdg-open", p])

    def _open_plots(self):
        try:
            d = pc.output_paths(self.get_config())["plots"]
        except Exception:
            d = None
        idx = Path(d) / "index.html" if d else None
        if idx and not idx.exists() and self.vars["output_root"].get():   # results of older versions
            old = Path(self.vars["output_root"].get()) / "analysis_plots" / "index.html"
            idx = old if old.exists() else idx
        if not idx or not idx.exists():
            return messagebox.showinfo("Plots", "No analysis plots yet -- run stage 5 first.")
        import webbrowser
        webbrowser.open(idx.resolve().as_uri())

    # ---------- running ----------
    def _log_line(self, line):
        tag = None
        if line.startswith(("✗", "⚠")) or "✗" in line[:40] or "Traceback" in line or "Error" in line[:20]:
            tag = "err"
        elif "✓" in line[:40]:
            tag = "ok"
        elif line.startswith(("===", "---")):
            tag = "head"
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run(self):
        cfg = self._check(quiet_ok=True)
        if cfg is None:
            return
        stages = [n for k, n in [("run_background", "background"), ("run_segmentation", "segmentation"),
                                 ("run_measurement", "measurement"), ("run_aggregation", "aggregation"),
                                 ("run_analysis", "analysis plots")] if cfg[k]]
        nc = (f"enabled (< {cfg['min_nuclei_confluent']} nuclei → '{cfg['foreground_mask']}' mask)"
              if cfg["nonconfluent_enabled"] else f"disabled (sparse scenes: {cfg['sparse_policy']})")
        msg = (f"Input:  {cfg['input_path']}\nOutput: {cfg['output_root']}\n\nStages: {', '.join(stages) or 'none'}\n"
               f"Background: {cfg['background_mode']}\nNon-confluent analysis: {nc}\n\nStart the analysis?")
        if not messagebox.askyesno("Run analysis", msg):
            return
        self._save_last()
        tmp = Path(tempfile.gettempdir()) / f"cell_analyzer_run_{os.getpid()}.json"
        pc.save_config_file(cfg, tmp)

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.nb.select(len(self.nb.tabs()) - 1)
        self.progress.configure(value=0, maximum=1)

        cmd = [sys.executable, "-u", "-m", "cell_analyzer.pipeline", "--config", str(tmp), "--no-prompt"]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        if os.name == "nt":
            # same as "conda activate": the environment's own DLL folders first, so Windows doesn't load
            # same-named DLLs from other programs (crash 0xC06D007F) when the window was started without it
            pre = sys.prefix
            dll_dirs = [pre, os.path.join(pre, "Library", "mingw-w64", "bin"), os.path.join(pre, "Library", "usr", "bin"),
                        os.path.join(pre, "Library", "bin"), os.path.join(pre, "Scripts"), os.path.join(pre, "bin")]
            env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])
            env.setdefault("CONDA_PREFIX", pre)
            env["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"
        kw = dict(cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                  env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(cmd, **kw)
        except Exception as e:
            return messagebox.showerror("Run", f"Could not start the analysis:\n{e}")
        self.run_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self.status.configure(text="Running…")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(("line", line.rstrip("\n")))
        proc.wait()
        self.q.put(("done", proc.returncode))

    def _poll_queue(self):
        try:
            for _ in range(500):
                kind, val = self.q.get_nowait()
                if kind == "line":
                    if val.startswith("[PROGRESS]"):
                        try:
                            _, stage, frac = val.split()
                            d, t = map(int, frac.split("/"))
                            self.progress.configure(maximum=max(t, 1), value=d)
                            self.status.configure(text=f"{stage}: {d}/{t}")
                        except ValueError:
                            pass
                    else:
                        self._log_line(val)
                else:
                    self._finished(val)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finished(self, code):
        self.proc = None
        self.run_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        if code == 0:
            self.progress.configure(value=self.progress["maximum"])
            self.status.configure(text="Finished ✓")
            self._log_line("=== Finished ===")
        else:
            self.status.configure(text=f"Stopped / failed (exit {code})")
            self._log_line(f"✗ Analysis process ended with exit code {code}")
            hint = explain_exit_code(code)
            if hint:
                self._log_line(f"⚠ {hint}")

    def _stop(self):
        if not self.proc:
            return
        if not messagebox.askyesno("Stop", "Stop the running analysis?\n(Finished scenes are kept; "
                                           "a restart skips them if overwrite is off.)"):
            return
        p = self.proc
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
                self.after(4000, lambda: p.poll() is None and subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(p.pid)]))
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)  # lets the log record 'cancelled'
                self.after(5000, lambda: p.poll() is None and os.killpg(os.getpgid(p.pid), signal.SIGKILL))
        except Exception as e:
            self._log_line(f"✗ could not stop cleanly: {e}")
        self.status.configure(text="Stopping…")

    def _on_close(self):
        if self.proc and not messagebox.askyesno("Quit", "An analysis is still running. Stop it and quit?"):
            return
        if self.proc:
            try:
                if os.name == "nt":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)])
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self._save_last()
        self.destroy()


if __name__ == "__main__":
    CellAnalyzerGUI().mainloop()
