#!/usr/bin/env python3
"""gui_restorer.py — шумодав-восстановитель ×1 (без смены размера).

Убирает шум/артефакты, правит текстуру после апскейла.
Чекпоинты: restorer_x1[_h256][_B..]_best/last.pth (больший base + _last первые).
"""
import os, re, glob, threading, time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from PIL import Image, ImageTk
import numpy as np
from restorer import RestorerMobile
from gui_realesr import enhance_realesr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_restorers(device):
    found = []
    for p in glob.glob("restorer_x1_*.pth"):
        m = re.match(r"restorer_x1(?:_h\d+)?(?:_B(\d+)(?:_(\d+))?)?(?:_W(\d+))?(?:_H(\d+))?_(last|best)\.pth$", os.path.basename(p))
        if not m:
            continue
        base = int(m.group(1) or 128)
        blocks = int(m.group(2) or 12)
        window = int(m.group(3) or 8)
        heads = int(m.group(4) or 4)
        kind = m.group(5)
        try:
            net = RestorerMobile(base=base, blocks=blocks, window=window, heads=heads).to(device)
            net.load_state_dict(torch.load(p, map_location=device), strict=False)
            net.eval()
            found.append((p, base, blocks, kind, net))
        except Exception as e:
            print(f"skip {p}: {e}")
    def key(c):
        p, base, blocks, kind, net = c
        return (base, blocks, 1 if '_h' in p else 0, 1 if kind == 'last' else 0)
    found.sort(key=key, reverse=True)
    return found


class RestorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Шумодав ×1 — очистка")
        self.root.geometry("1080x820")
        self.input_path = None
        self.tile_size = tk.IntVar(value=256)
        self.overlap = tk.IntVar(value=32)
        self.strength = tk.IntVar(value=100)   # сила чистки, %
        self.device = DEVICE
        self.model = None
        self.loaded = ""
        found = find_restorers(self.device)
        if found:
            p, base, blocks, kind, net = found[0]
            self.model = net
            self.loaded = os.path.basename(p)
        self.create_widgets()
        self.model_status.config(text=self.loaded if self.loaded else "нет чекпоинта — обучите",
                                 fg="green" if self.loaded else "red")

    def create_widgets(self):
        tk.Label(self.root, text="Шумодав-восстановитель ×1", font=("Segoe UI", 16, "bold"), pady=10).pack()
        main = tk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        top = tk.Frame(main); top.pack(fill=tk.X, pady=6)
        tk.Button(top, text="📂 Выбрать фото", command=self.select_file, bg="#4CAF50",
                  fg="white", padx=18, pady=8, relief=tk.FLAT).pack(side=tk.LEFT)
        self.file_label = tk.Label(top, text="Файл не выбран", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=12)
        self.model_status = tk.Label(top, text=""); self.model_status.pack(side=tk.RIGHT)
        settings = tk.LabelFrame(main, text="⚙️", padx=12, pady=10); settings.pack(fill=tk.X, pady=6)
        tk.Label(settings, text="Тайл:").grid(row=0, column=0, sticky=tk.W)
        tk.Spinbox(settings, from_=128, to=512, increment=64,
                   textvariable=self.tile_size, width=8).grid(row=0, column=1, padx=8)
        tk.Label(settings, text="Сила:").grid(row=0, column=2, sticky=tk.W)
        tk.Scale(settings, from_=0, to=100, orient=tk.HORIZONTAL,
                 variable=self.strength, length=160).grid(row=0, column=3, padx=8)
        self.clean_btn = tk.Button(settings, text="🧹 ОЧИСТИТЬ", command=self.start_clean,
                                   bg="#2196F3", fg="white", padx=26, pady=8,
                                   relief=tk.FLAT, font=("Segoe UI", 11, "bold"))
        self.clean_btn.grid(row=0, column=4, padx=(24, 0))
        prog = tk.LabelFrame(main, text="📊", padx=12, pady=10); prog.pack(fill=tk.X, pady=6)
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=4)
        self.status_label = tk.Label(prog, text="Ожидание…"); self.status_label.pack()
        prev = tk.Frame(main); prev.pack(fill=tk.BOTH, expand=True, pady=8)
        self.before_label = tk.Label(prev, text="До", bg="#eee", width=40, height=12)
        self.before_label.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        self.after_label = tk.Label(prev, text="После", bg="#eee", width=40, height=12)
        self.after_label.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        self.is_processing = False

    def select_file(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")])
        if not p:
            return
        self.input_path = p
        self.file_label.config(text=os.path.basename(p))
        img = Image.open(p).convert("RGB")
        thumb = img.copy(); thumb.thumbnail((480, 380))
        photo = ImageTk.PhotoImage(thumb)
        self.before_label.config(image=photo, text=""); self.before_label.image = photo

    def start_clean(self):
        if not self.input_path or self.is_processing:
            return
        if self.model is None:
            messagebox.showerror("Ошибка", "Нет модели — обучите")
            return
        self.is_processing = True; self.clean_btn.config(state=tk.DISABLED, text="⏳...")
        self.progress_var.set(0); self.status_label.config(text="Старт…")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            out_path = str(Path(self.input_path).with_suffix("")) + "_clean.jpg"
            img_np = np.array(Image.open(self.input_path).convert("RGB")).astype(np.float32) / 255
            def cb(msg, pct):
                self.root.after(0, lambda: (self.progress_var.set(pct), self.status_label.config(text=msg)))
            t0 = time.time()
            tile = max(64, (self.tile_size.get() // 16) * 16)
            out = enhance_realesr(self.model, img_np, scale=1, tile=tile,
                                  overlap=self.overlap.get(), device=self.device, progress=cb)
            # сила чистки: подмес исходника (0% = bypass, 100% = полная)
            a = self.strength.get() / 100.0
            if a < 1.0:
                out = (np.clip((1.0 - a) * img_np + a * (out.astype(np.float32) / 255), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(out).save(out_path, quality=95)
            elapsed = time.time() - t0
            preview = Image.fromarray(out); preview.thumbnail((480, 380))
            photo = ImageTk.PhotoImage(preview)
            def done():
                self.after_label.config(image=photo, text=""); self.after_label.image = photo
                self.status_label.config(text=f"✅ {Path(out_path).name} {elapsed:.1f}с")
                self.progress_var.set(100)
            self.root.after(0, done)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.clean_btn.config(state=tk.NORMAL, text="🧹 ОЧИСТИТЬ"))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    RestorerApp(root).run()
