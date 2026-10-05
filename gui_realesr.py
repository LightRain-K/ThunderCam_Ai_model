#!/usr/bin/env python3
"""gui_realesr.py — тест RealESR-Mobile ×2/×4"""

import os, re, glob, threading, time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from PIL import Image, ImageTk
import numpy as np
from realesr_mobile import RealESRMobile

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_models(scale, device):
    """Все чекпоинты ×scale, лучший первый: big base -> h-hr -> _last -> _best."""
    found = []
    for p in glob.glob(f"realesr_mobile_x{scale}_*.pth"):
        m = re.match(r"realesr_mobile_x(\d+)(?:_h\d+)?(?:_B(\d+)(?:_(\d+))?)?(?:_W(\d+))?(?:_H(\d+))?_(last|best)\.pth$", os.path.basename(p))
        if not m or int(m.group(1)) != scale:
            continue
        base = int(m.group(2) or 128)
        blocks = int(m.group(3) or 12)
        window = int(m.group(4) or 8)
        heads = int(m.group(5) or 4)
        kind = m.group(6)
        try:
            net = RealESRMobile(scale=scale, base=base, blocks=blocks, window=window, heads=heads).to(device)
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

def enhance_realesr(model, img_np, scale=2, tile=256, overlap=32, device='cuda', progress=None):
    h_img, w_img, c = img_np.shape
    # зеркальный паддинг — иначе угол картинки попадает под 0 Hanning-окна
    # тайла и размазывается (артефакт блендинга, не модели)
    pad = overlap
    img = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    h, w = img.shape[0], img.shape[1]
    # LR уже scale раз меньше HR, модель делает ×scale
    # тайлим LR
    out_h, out_w = h*scale, w*scale
    result = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    win1d = np.hanning(tile*scale)
    win2d = np.outer(win1d, win1d)[:,:,None]
    step = tile - overlap
    ys = list(range(0, max(h - overlap,1), step))
    xs = list(range(0, max(w - overlap,1), step))
    if ys[-1]+tile < h: ys.append(max(0, h-tile))
    if xs[-1]+tile < w: xs.append(max(0, w-tile))
    ys, xs = sorted(set(ys)), sorted(set(xs))
    total = len(ys)*len(xs)
    cnt=0
    for y in ys:
        for x in xs:
            th, tw = min(tile, h-y), min(tile, w-x)
            patch = img[y:y+th, x:x+tw]
            if th<tile or tw<tile:
                tmp=np.zeros((tile,tile,c), dtype=np.float32)
                tmp[:th,:tw]=patch
                patch=tmp
            t=torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device=='cuda')):
                out=model(t).squeeze(0).permute(1,2,0).cpu().numpy()
            oh, ow = th*scale, tw*scale
            out=out[:oh, :ow]
            wgt=win2d[:oh, :ow]
            result[y*scale:y*scale+oh, x*scale:x*scale+ow] += out*wgt
            weight[y*scale:y*scale+oh, x*scale:x*scale+ow] += wgt
            cnt+=1
            if progress: progress(f"{cnt}/{total}", int(cnt/total*90)+5)
    result=np.clip(result/np.maximum(weight,1e-6),0,1)
    # срезаем паддинг обратно
    result = result[pad*scale:(pad+h_img)*scale, pad*scale:(pad+w_img)*scale]
    return (result*255).astype(np.uint8)

class RealesrApp:
    def __init__(self, root):
        self.root=root
        self.root.title("RealESR-Mobile ×2/×4 — тест")
        self.root.geometry("1080x820")
        self.input_path=None
        self.cascade=tk.BooleanVar(value=True)   # ×4 = ×2 + ×2 (честные детали)
        self.tile_size=tk.IntVar(value=256)
        self.overlap=tk.IntVar(value=32)
        self.device=DEVICE
        self.models={}
        self.loaded_desc=[]
        for s in [2,4]:
            found = find_models(s, self.device)
            if found:
                p, base, blocks, kind, net = found[0]
                self.models[s]=net
                self.loaded_desc.append(f"×{s}:{os.path.basename(p)}")
        self.scale=tk.IntVar(value=4 if 4 in self.models else 2)
        self.create_widgets()
        self.scale.set(4 if 4 in self.models else 2)
        self.check_model()

    def check_model(self):
        txt=", ".join(self.loaded_desc) if self.loaded_desc else "нет чекпоинта — обучите"
        self.model_status.config(text=txt, fg="green" if self.loaded_desc else "red")

    def create_widgets(self):
        tk.Label(self.root, text="RealESR-Mobile — тест ×2/×4", font=("Segoe UI",16,"bold"), pady=10).pack()
        main=tk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        top=tk.Frame(main); top.pack(fill=tk.X, pady=6)
        tk.Button(top, text="📂 Выбрать фото", command=self.select_file, bg="#4CAF50", fg="white", padx=18, pady=8, relief=tk.FLAT).pack(side=tk.LEFT)
        self.file_label=tk.Label(top, text="Файл не выбран", fg="gray"); self.file_label.pack(side=tk.LEFT, padx=12)
        self.model_status=tk.Label(top, text=""); self.model_status.pack(side=tk.RIGHT)
        settings=tk.LabelFrame(main, text="⚙️", padx=12, pady=10); settings.pack(fill=tk.X, pady=6)
        tk.Label(settings, text="Scale:").grid(row=0,column=0,sticky=tk.W)
        tk.Radiobutton(settings, text="×2", variable=self.scale, value=2).grid(row=0,column=1)
        tk.Radiobutton(settings, text="×4", variable=self.scale, value=4).grid(row=0,column=2)
        self.cascade_chk=tk.Checkbutton(settings, text="каскад ×2+×2", variable=self.cascade)
        self.cascade_chk.grid(row=0,column=3)
        tk.Label(settings, text="Тайл:").grid(row=0,column=4,sticky=tk.W,padx=(16,0))
        tk.Spinbox(settings, from_=128,to=512,increment=64,textvariable=self.tile_size,width=8).grid(row=0,column=5,padx=8)
        self.enhance_btn=tk.Button(settings, text="🚀 УВЕЛИЧИТЬ", command=self.start_enhance, bg="#FF5722", fg="white", padx=26, pady=8, relief=tk.FLAT, font=("Segoe UI",11,"bold"))
        self.enhance_btn.grid(row=0,column=6,padx=(24,0))
        prog=tk.LabelFrame(main, text="📊", padx=12, pady=10); prog.pack(fill=tk.X, pady=6)
        self.progress_var=tk.DoubleVar()
        ttk.Progressbar(prog, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=4)
        self.status_label=tk.Label(prog, text="Ожидание…"); self.status_label.pack()
        prev=tk.Frame(main); prev.pack(fill=tk.BOTH, expand=True, pady=8)
        self.before_label=tk.Label(prev, text="До", bg="#eee", width=40, height=12); self.before_label.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        self.after_label=tk.Label(prev, text="После", bg="#eee", width=40, height=12); self.after_label.pack(side=tk.LEFT, padx=6, fill=tk.BOTH, expand=True)
        self.is_processing=False

    def select_file(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png"),("All","*.*")])
        if not p: return
        self.input_path=p
        self.file_label.config(text=os.path.basename(p))
        img=Image.open(p).convert("RGB")
        thumb=img.copy(); thumb.thumbnail((480,380))
        photo=ImageTk.PhotoImage(thumb)
        self.before_label.config(image=photo, text=""); self.before_label.image=photo

    def start_enhance(self):
        if not self.input_path or self.is_processing: return
        scale=self.scale.get()
        use_cascade = scale==4 and self.cascade.get() and 2 in self.models
        need = 2 if use_cascade else scale
        if need not in self.models:
            messagebox.showerror("Ошибка", f"Нет модели ×{need} — обучите")
            return
        self.is_processing=True; self.enhance_btn.config(state=tk.DISABLED, text="⏳...")
        self.progress_var.set(0); self.status_label.config(text="Старт…")
        threading.Thread(target=self._run, args=(scale,), daemon=True).start()

    def get_model(self, s):
        return self.models.get(s, None)

    def _run(self, scale):
        try:
            out_path=str(Path(self.input_path).with_suffix(""))+f"_x{scale}.jpg"
            img_np=np.array(Image.open(self.input_path).convert("RGB")).astype(np.float32)/255
            def cb(msg,pct):
                self.root.after(0, lambda: (self.progress_var.set(pct), self.status_label.config(text=msg)))
            t0=time.time()
            tile=max(64, (self.tile_size.get()//16)*16)   # окно 8/16: тайл кратен 16
            use_cascade = scale==4 and self.cascade.get() and 2 in self.models
            if use_cascade:
                # ×4 = ×2 + ×2 — каждый шаг честный, детали точнее
                mid=enhance_realesr(self.get_model(2), img_np, scale=2,
                                    tile=tile, overlap=self.overlap.get(),
                                    device=self.device, progress=cb)
                out=enhance_realesr(self.get_model(2), mid.astype(np.float32)/255, scale=2,
                                    tile=tile, overlap=self.overlap.get(),
                                    device=self.device, progress=cb)
            else:
                m=self.get_model(scale)
                if m is None:
                    raise RuntimeError(f"Нет модели ×{scale}")
                out=enhance_realesr(m, img_np, scale=scale, tile=tile,
                                    overlap=self.overlap.get(), device=self.device, progress=cb)
            Image.fromarray(out).save(out_path, quality=95)
            elapsed=time.time()-t0
            preview=Image.fromarray(out); preview.thumbnail((480,380))
            photo=ImageTk.PhotoImage(preview)
            def done():
                self.after_label.config(image=photo, text=""); self.after_label.image=photo
                self.status_label.config(text=f"✅ {Path(out_path).name} {elapsed:.1f}с ×{scale}")
                self.progress_var.set(100)
            self.root.after(0, done)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.is_processing=False
            self.root.after(0, lambda: self.enhance_btn.config(state=tk.NORMAL, text="🚀 УВЕЛИЧИТЬ"))

    def run(self):
        self.root.mainloop()

if __name__=="__main__":
    root=tk.Tk()
    RealesrApp(root).run()
