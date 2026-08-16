#!/usr/bin/env python3
"""
MouseTrack — Desktop GUI for multi-object mouse video segmentation using SAM3.

Design: "Neural Lab" — palette derived from confocal microscopy fluorescence colors.
The same cyan/green/magenta/yellow used for object masks define the entire UI.
"""

import time
_t0 = time.time()

_LOADING_STEPS = 6
_loading_step = 0

def _loading_bar(label, done=False):
    """Prints a simple text progress bar in the console while the heavy
    libraries (torch, transformers...) are imported, before the Qt window
    can even exist. Percentage is step-based (1 of 6, 2 of 6, ...), not
    time-based, since import time varies a lot machine to machine."""
    global _loading_step
    if not done:
        _loading_step += 1
    pct = int(_loading_step / _LOADING_STEPS * 100)
    filled = int(pct / 5)
    bar = "#" * filled + "-" * (20 - filled)
    elapsed = time.time() - _t0
    end = "\n" if _loading_step >= _LOADING_STEPS else ""
    print(f"\r[LOADING] [{bar}] {pct:3d}%  {label:<28} ({elapsed:5.1f}s)", end=end, flush=True)

_loading_bar("Starting…", done=True)

import os, sys, json, glob, subprocess, base64
_loading_bar("Standard library")

import numpy as np
_loading_bar("NumPy")

import cv2
_loading_bar("OpenCV")

import torch
_loading_bar(f"PyTorch (CUDA: {torch.cuda.is_available()})")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QLineEdit, QSpinBox, QProgressBar, QPlainTextEdit, QFileDialog,
    QStackedWidget, QGroupBox, QSizePolicy, QMessageBox, QRadioButton,
    QButtonGroup, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush,
    QLinearGradient, QPainterPath, QRadialGradient, QFontDatabase
)
_loading_bar("PyQt5")

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator
_loading_bar("Transformers + Accelerate")

# Embedded MouseTrack icon (mouse silhouette, recolored to match the header
# cyan text exactly) as base64 PNG, so no external asset file is needed.
MOUSE_ICON_B64 = "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"


# ═══════════════════════════ PALETTE ═══════════════════════════════════
# Fluorescence microscopy–inspired: deep black + fluorescent accents
BG          = "#050a0e"   # microscope black
PANEL       = "#0b1622"   # deep tissue
SURFACE     = "#0f2233"   # elevated surface
SURFACE_H   = "#142a40"   # hover surface
BORDER      = "#1a3a50"   # default border
BORDER_A    = "#00c8e0"   # active border (cyan)
CYAN        = "#00c8e0"   # primary accent — cyan fluorophore / mask obj1
CYAN_DARK   = "#008fa0"
CYAN_GLOW   = "#00e5ff"
GREEN       = "#00e676"   # success — GFP green / mask obj3
GREEN_DARK  = "#00a857"
MAGENTA     = "#ff4081"   # danger — TRITC magenta / mask obj4
MAGENTA_DK  = "#c41162"
YELLOW      = "#ffd740"   # warning / mask obj2
TEXT        = "#dce8f0"   # primary text
TEXT_S      = "#6a8fa8"   # secondary text
TEXT_M      = "#2d5066"   # muted text
TERMINAL    = "#00ffb2"   # log terminal color
# ═══════════════════════════════════════════════════════════════════════

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
CHUNK_SIZE = 500
FLUSH_EVERY_N_FRAMES = 1000
MASK_ALPHA = 0.5

OBJ_COLORS_BGR = {1:(255,255,0), 2:(0,255,255), 3:(0,255,0), 4:(255,0,255)}
OBJ_COLORS_RGB = {oid:(r,g,b) for oid,(b,g,r) in OBJ_COLORS_BGR.items()}
OBJ_COLORS_QT  = {
    1: QColor(0,200,224),   # cyan
    2: QColor(255,215,64),  # yellow
    3: QColor(0,230,118),   # green
    4: QColor(255,64,129),  # magenta
}

_model = _processor = _device = None

def load_model_global(log_fn=None):
    global _model, _processor, _device
    if _model is None:
        if log_fn: log_fn("Loading SAM3 model…")
        _device = Accelerator().device
        _model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(_device, dtype=torch.bfloat16)
        _processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        # torch.compile disabilitato: incompatibile con le operazioni di reshape
        # degli embeddings di posizione in SAM3 (RuntimeError con backend inductor).
        if log_fn: log_fn(f"Model ready on {_device}.")
    return _model, _processor, _device

def base_name(p): return os.path.splitext(os.path.basename(p))[0]

def get_output_paths(video_path, out_dir, start_frame=0):
    bn = base_name(video_path)
    sfx = f"_resume{start_frame:07d}" if start_frame > 0 else ""
    return (os.path.join(out_dir, f"{bn}_output{sfx}.mp4"),
            os.path.join(out_dir, f"{bn}_shards{sfx}"))

def get_merged_paths(video_path, out_dir):
    bn = base_name(video_path)
    return (os.path.join(out_dir, f"{bn}_merged.mp4"),
            os.path.join(out_dir, f"{bn}_shards_merged"))

def batch_status_path(out_dir): return os.path.join(out_dir, "batch_status.json")

def load_batch_status(out_dir):
    p = batch_status_path(out_dir)
    return json.load(open(p)) if os.path.exists(p) else {}

def save_batch_status(out_dir, s):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(s, open(batch_status_path(out_dir),"w"), indent=2)

def add_segment(out_dir, video_path, start_frame, vid_out, shd_out):
    s = load_batch_status(out_dir); bn = os.path.basename(video_path)
    e = s.setdefault(bn, {"status":"pending","segments":[]})
    e["segments"].append({"start_frame":start_frame,"video":vid_out,"shards":shd_out})
    e["status"] = "done" if len(e["segments"])==1 and start_frame==0 else "has_resume"
    save_batch_status(out_dir, s)

def mark_merged(out_dir, video_path, mv, ms):
    s = load_batch_status(out_dir); bn = os.path.basename(video_path)
    s[bn].update({"status":"merged","merged_video":mv,"merged_shards":ms})
    save_batch_status(out_dir, s)

# ── Annotation persistence ─────────────────────────────────────────────
def annotations_path(out_dir): return os.path.join(out_dir, "annotations.json")

def load_annotations(out_dir):
    p = annotations_path(out_dir)
    return json.load(open(p)) if os.path.exists(p) else {}

def save_annotation(out_dir, video_path, start_frame, points):
    """Save click points for a video so propagation can run later without re-clicking."""
    os.makedirs(out_dir, exist_ok=True)
    ann = load_annotations(out_dir)
    ann[os.path.basename(video_path)] = {"start_frame": start_frame, "points": points}
    json.dump(ann, open(annotations_path(out_dir),"w"), indent=2)

def delete_annotation(out_dir, video_path):
    ann = load_annotations(out_dir)
    ann.pop(os.path.basename(video_path), None)
    json.dump(ann, open(annotations_path(out_dir),"w"), indent=2)


# ═══════════════════════════ WORKERS ═══════════════════════════════════

class PropagationThread(QThread):
    progress      = pyqtSignal(int, int)
    log           = pyqtSignal(str)
    finished      = pyqtSignal(str, str, bool)
    overlap_alert = pyqtSignal(int, float)

    def __init__(self, video_path, points, start_frame, out_dir):
        super().__init__()
        self.video_path = video_path; self.points = points
        self.start_frame = start_frame; self.out_dir = out_dir
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            model, processor, device = load_model_global(self.log.emit)
            sf = self.start_frame
            vid_out, shd_out = get_output_paths(self.video_path, self.out_dir, sf)
            os.makedirs(shd_out, exist_ok=True)

            obj_ids = sorted(set(p[0] for p in self.points))
            pts_by_obj = {oid:[] for oid in obj_ids}
            for oid,x,y in self.points: pts_by_obj[oid].append((x,y))
            orig_pts = {oid:list(v) for oid,v in pts_by_obj.items()}

            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS)
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            to_proc = total - sf
            if sf > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(vid_out, fourcc, fps, (W,H))
            buf = {}; shard_idx = 0; obj_cents = {oid:[] for oid in obj_ids}
            gfi = 0; chunk_num = 0

            def flush(b, si):
                if not b: return {}, si
                np.savez_compressed(os.path.join(shd_out,f"shard_{si:05d}.npz"),**b)
                return {}, si+1

            def read_chunk(cap, n):
                out=[]
                for _ in range(n):
                    ok,f = cap.read()
                    if not ok: break
                    out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
                return out

            while gfi < to_proc and not self._cancel:
                n = min(CHUNK_SIZE, to_proc-gfi)
                chunk = read_chunk(cap, n)
                if not chunk: break
                cl = len(chunk)

                sess = processor.init_video_session(video=chunk,
                    inference_device=device, processing_device="cpu",
                    video_storage_device="cpu", dtype=torch.bfloat16)

                if chunk_num == 0:
                    seed_ids = list(obj_ids)
                    seed_pts = {oid:list(pts_by_obj[oid]) for oid in obj_ids}
                else:
                    spl=[]
                    for oid in obj_ids:
                        if obj_cents.get(oid):
                            _,cx,cy = obj_cents[oid][-1]; spl.append((oid,cx,cy))
                    if not spl:
                        self.log.emit(f"[WARNING] chunk {chunk_num}: tracking lost, re-seeding from original clicks.")
                        spl = [(oid,x,y) for oid in obj_ids for x,y in orig_pts[oid]]
                    seed_ids = sorted(set(o for o,_,_ in spl))
                    seed_pts = {oid:[] for oid in seed_ids}
                    for oid,x,y in spl: seed_pts[oid].append((x,y))

                ip = [[[[float(x),float(y)] for x,y in seed_pts[oid]] for oid in seed_ids]]
                il = [[[1]*len(seed_pts[oid]) for oid in seed_ids]]
                processor.add_inputs_to_inference_session(sess,frame_idx=0,
                    obj_ids=list(seed_ids),input_points=ip,input_labels=il)
                self.log.emit(f"Chunk {chunk_num} — frames {sf+gfi}…{sf+gfi+cl-1}  seeds: {seed_ids}")

                with torch.no_grad(): model(inference_session=sess,frame_idx=0)

                for out in model.propagate_in_video_iterator(sess):
                    if self._cancel: break
                    li = out.frame_idx; gi = sf+gfi+li
                    masks_r = processor.post_process_masks([out.pred_masks],
                        original_sizes=[[H,W]],binarize=True)[0]
                    sids = list(sess.obj_ids)
                    scores = getattr(out,"iou_scores",None)
                    if scores is not None and torch.is_tensor(scores):
                        scores=scores.squeeze().tolist()
                        if not isinstance(scores,list): scores=[scores]

                    raw={}
                    for i,oid in enumerate(sids):
                        m=masks_r[i]
                        m=m.cpu().numpy().astype(bool) if torch.is_tensor(m) else np.asarray(m,bool)
                        if m.ndim>2: m=m.squeeze()
                        if m.shape!=(H,W): m=cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
                        raw[oid]=m

                    if len(sids)>1:
                        for a in range(len(sids)):
                            for b in range(a+1,len(sids)):
                                oa,ob=sids[a],sids[b]
                                ov=raw[oa]&raw[ob]
                                if not ov.any(): continue
                                aA,aB=int(raw[oa].sum()),int(raw[ob].sum())
                                iou=ov.sum()/(aA+aB-ov.sum()+1e-9)
                                if iou>0.3: self.overlap_alert.emit(gi,float(iou))
                                sa=(scores[a] if scores and a<len(scores) else aA)
                                sb=(scores[b] if scores and b<len(scores) else aB)
                                if sa>=sb: raw[ob]=raw[ob]&~ov
                                else:      raw[oa]=raw[oa]&~ov

                    frame_bgr=cv2.cvtColor(chunk[li],cv2.COLOR_RGB2BGR)
                    ovl=frame_bgr.copy(); cl_l=np.zeros_like(frame_bgr); any_m=np.zeros((H,W),bool)
                    for oid in sids:
                        m=raw[oid]
                        buf[f"frame{gi:07d}_obj{oid}_mask_packed"]=np.packbits(m)
                        buf[f"frame{gi:07d}_obj{oid}_mask_shape"]=np.array(m.shape,dtype=np.int32)
                        ys,xs=np.where(m)
                        if len(xs):
                            cx,cy=xs.mean(),ys.mean()
                            obj_cents.setdefault(oid,[]).append((gi,cx,cy))
                            cl_l[m]=OBJ_COLORS_BGR.get(oid,(255,255,255))
                            any_m|=m
                            cv2.putText(ovl,str(oid),(int(cx),int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,OBJ_COLORS_BGR.get(oid,(255,255,255)),2)
                    ovl[any_m]=(ovl[any_m]*(1-MASK_ALPHA)+cl_l[any_m]*MASK_ALPHA).astype(np.uint8)
                    writer.write(ovl)
                    if (gi+1)%FLUSH_EVERY_N_FRAMES==0: buf,shard_idx=flush(buf,shard_idx)
                    self.progress.emit(gfi+li+1, to_proc)

                del sess
                if device.type=="cuda": torch.cuda.empty_cache()
                gfi+=cl; chunk_num+=1

            buf,shard_idx=flush(buf,shard_idx)
            cap.release(); writer.release()
            np.savez_compressed(os.path.join(shd_out,"centroids.npz"),
                **{f"centroids_obj{oid}":np.array(v) for oid,v in obj_cents.items()})
            self.finished.emit(vid_out,shd_out, not self._cancel)
        except Exception as e:
            import traceback; self.log.emit(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.finished.emit("","",False)


class PreviewThread(QThread):
    done = pyqtSignal(object, str)
    def __init__(self, frame_rgb, points): super().__init__(); self.frame_rgb=frame_rgb; self.points=points
    def run(self):
        try:
            model,processor,device=load_model_global()
            frgb=self.frame_rgb; H,W=frgb.shape[:2]
            obj_ids=sorted(set(p[0] for p in self.points))
            pts={oid:[] for oid in obj_ids}
            for oid,x,y in self.points: pts[oid].append((x,y))
            sess=processor.init_video_session(video=[frgb],inference_device=device,
                processing_device="cpu",video_storage_device="cpu",dtype=torch.bfloat16)
            ip=[[[[float(x),float(y)] for x,y in pts[oid]] for oid in obj_ids]]
            il=[[[1]*len(pts[oid]) for oid in obj_ids]]
            processor.add_inputs_to_inference_session(sess,frame_idx=0,obj_ids=obj_ids,input_points=ip,input_labels=il)
            with torch.no_grad(): out=model(inference_session=sess,frame_idx=0)
            masks=processor.post_process_masks([out.pred_masks],original_sizes=[[H,W]],binarize=True)[0]
            sids=list(sess.obj_ids)
            ovl=frgb.copy(); cl=np.zeros_like(ovl); any_m=np.zeros((H,W),bool); dbg=[]
            for i,oid in enumerate(sids):
                m=masks[i]; m=m.cpu().numpy().astype(bool) if torch.is_tensor(m) else np.asarray(m,bool)
                if m.ndim>2: m=m.squeeze()
                if m.shape!=(H,W): m=cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
                cl[m]=OBJ_COLORS_RGB.get(oid,(255,255,255)); any_m|=m
                dbg.append(f"Object {oid}: {int(m.sum())} px")
            ovl[any_m]=(ovl[any_m]*(1-MASK_ALPHA)+cl[any_m]*MASK_ALPHA).astype(np.uint8)
            del sess
            if device.type=="cuda": torch.cuda.empty_cache()
            self.done.emit(ovl,"\n".join(dbg))
        except Exception as e:
            import traceback; self.done.emit(None,f"Preview error: {e}")


class MergeThread(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)
    def __init__(self, video_path, segments, out_dir):
        super().__init__(); self.video_path=video_path; self.segments=segments; self.out_dir=out_dir
    def run(self):
        try:
            mv,ms=get_merged_paths(self.video_path,self.out_dir)
            segs=sorted(self.segments,key=lambda s:s["start_frame"])
            swaps=[s["start_frame"] for s in segs[1:]]

            self.log.emit("Merging video segments via ffmpeg…")
            fparts=[]; inputs=[]
            for i,seg in enumerate(segs):
                inputs+=["-i",seg["video"]]
                if i==0 and swaps:
                    fparts.append(f"[0:v]trim=start_frame=0:end_frame={swaps[0]},setpts=PTS-STARTPTS[v0]")
                else:
                    fparts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            conc="".join(f"[v{i}]" for i in range(len(segs)))
            fstr=";".join(fparts)+f";{conc}concat=n={len(segs)}:v=1:a=0[outv]"
            cmd=["ffmpeg","-y"]+inputs+["-filter_complex",fstr,"-map","[outv]",
                 "-c:v","libx264","-crf","20","-preset","ultrafast",mv]
            res=subprocess.run(cmd,capture_output=True,text=True)
            if res.returncode!=0: self.log.emit(f"ffmpeg error:\n{res.stderr}"); self.finished.emit(False,"",""); return
            self.log.emit(f"Video merged → {mv}")

            self.log.emit("Merging mask shards…")
            os.makedirs(ms,exist_ok=True)
            buf={}; bframes=set(); oi=0
            def flush_m():
                nonlocal buf,bframes,oi
                if not buf: return
                np.savez_compressed(os.path.join(ms,f"shard_{oi:05d}.npz"),**buf)
                buf={}; bframes=set(); oi+=1

            # Detect gaps between segments (frames where no segment has masks)
            # Gap = [end_of_seg_i .. start_of_seg_{i+1})
            # For each gap frame, write empty masks (all-zero packed bits) for
            # all object IDs so that every frame 0..total_frames has a key in
            # the merged shards — the downstream DLC-correction pipeline can
            # then rely on centroid interpolation instead of crashing on missing keys.
            obj_ids_known = set()
            frame_H = frame_W = None  # will be read from the first valid shard

            # First pass: collect obj_ids and frame shape from existing shards
            for seg in segs:
                for sf in sorted(glob.glob(os.path.join(seg["shards"],"shard_*.npz")))[:1]:
                    d=np.load(sf)
                    for k in d.files:
                        if "_mask_shape" in k:
                            oid=int(k.split("_obj")[1].split("_")[0])
                            obj_ids_known.add(oid)
                            if frame_H is None:
                                sh=d[k]; frame_H,frame_W=int(sh[0]),int(sh[1])
                    if frame_H is not None: break

            if frame_H is None: frame_H,frame_W=1,1  # fallback, should never happen

            def write_empty_mask(fidx, oid):
                """Write an all-zero packed mask for a gap frame."""
                empty=np.zeros(frame_H*frame_W,dtype=bool)
                buf[f"frame{fidx:07d}_obj{oid}_mask_packed"]=np.packbits(empty)
                buf[f"frame{fidx:07d}_obj{oid}_mask_shape"]=np.array([frame_H,frame_W],dtype=np.int32)
                buf[f"frame{fidx:07d}_obj{oid}_mask_gap"]=np.array([1],dtype=np.uint8)  # flag: interpolated frame
                bframes.add(fidx)

            # Build list of (fmin, fmax) for each segment
            seg_ranges=[]
            for si,seg in enumerate(segs):
                fmin=segs[si]["start_frame"] if si>0 else 0
                fmax=swaps[si] if si<len(swaps) else float("inf")
                seg_ranges.append((fmin,fmax))

            # Identify gaps: between end of seg i and start of seg i+1
            # end of seg i = swaps[i-1] (exclusive) from seg i's perspective
            # start of seg i+1 = segs[i+1]["start_frame"]
            gaps=[]  # list of (gap_start_frame, gap_end_frame_exclusive)
            for i in range(len(segs)-1):
                gap_start=swaps[i]       # first frame NOT in seg i
                gap_end=segs[i+1]["start_frame"]   # first frame IN seg i+1
                if gap_end>gap_start:
                    gaps.append((gap_start, gap_end))
                    self.log.emit(f"  Gap detected: frames {gap_start}–{gap_end-1} ({gap_end-gap_start} frames) — will insert empty masks")

            # Second pass: write real masks from each segment, then fill gaps
            for si,seg in enumerate(segs):
                fmin,fmax=seg_ranges[si]
                for sf in sorted(glob.glob(os.path.join(seg["shards"],"shard_*.npz"))):
                    d=np.load(sf)
                    for k in d.files:
                        if "_mask_" not in k: continue
                        fidx=int(k.split("_obj")[0].replace("frame",""))
                        if fmin<=fidx<fmax:
                            buf[k]=d[k]
                            if "_mask_packed" in k: bframes.add(fidx)
                    if len(bframes)>=FLUSH_EVERY_N_FRAMES: flush_m()
                    self.log.emit(f"  {os.path.basename(sf)} (seg {si+1}/{len(segs)})")

                # Fill the gap after this segment (if any)
                if si < len(gaps):
                    g_start,g_end=gaps[si]
                    self.log.emit(f"  Filling gap frames {g_start}–{g_end-1} with empty masks…")
                    for gf in range(g_start, g_end):
                        for oid in obj_ids_known:
                            write_empty_mask(gf, oid)
                        if len(bframes)>=FLUSH_EVERY_N_FRAMES: flush_m()
            flush_m()

            self.log.emit("Merging and interpolating centroids…")
            ac={}
            for si,seg in enumerate(segs):
                cp=os.path.join(seg["shards"],"centroids.npz")
                if not os.path.exists(cp): continue
                fmin,fmax=seg_ranges[si]
                d=np.load(cp)
                for k in d.files:
                    arr=d[k]; mask=(arr[:,0]>=fmin)&(arr[:,0]<fmax)
                    ac.setdefault(k,[]).append(arr[mask])

            # Concatenate real centroids from all segments
            mc={}
            for k,parts in ac.items():
                merged=np.concatenate(parts)
                mc[k]=merged[merged[:,0].argsort()]

            # Linear interpolation of centroids across each gap
            # For gap frames we add rows (frame_idx, cx_interp, cy_interp)
            # so that the downstream script has a position estimate for every frame.
            for gi,(g_start,g_end) in enumerate(gaps):
                for k in mc:
                    arr=mc[k]
                    # Last real centroid before the gap
                    before=arr[arr[:,0]<g_start]
                    # First real centroid after the gap
                    after=arr[arr[:,0]>=g_end]
                    if len(before)==0 or len(after)==0:
                        continue  # can't interpolate without both endpoints
                    f0,cx0,cy0=before[-1]
                    f1,cx1,cy1=after[0]
                    gap_frames=np.arange(g_start,g_end)
                    # Linear interpolation between (f0,cx0,cy0) and (f1,cx1,cy1)
                    t=(gap_frames-f0)/(f1-f0)
                    cx_interp=cx0+(cx1-cx0)*t
                    cy_interp=cy0+(cy1-cy0)*t
                    interp_rows=np.column_stack([gap_frames,cx_interp,cy_interp])
                    # Insert and re-sort
                    mc[k]=np.concatenate([arr,interp_rows])
                    mc[k]=mc[k][mc[k][:,0].argsort()]
                    self.log.emit(f"  Interpolated {len(gap_frames)} centroid(s) for gap {g_start}–{g_end-1} [{k}]")

            np.savez_compressed(os.path.join(ms,"centroids.npz"),**mc)

            # Save a gap report so the downstream DLC-correction script
            # knows exactly which frames were interpolated vs real
            if gaps:
                report={"gaps":[{"start":int(a),"end_exclusive":int(b),"n_frames":int(b-a)} for a,b in gaps]}
                json.dump(report, open(os.path.join(ms,"gap_report.json"),"w"), indent=2)
                self.log.emit(f"  Gap report saved → {os.path.join(ms,'gap_report.json')}")

            self.log.emit("All done!"); self.finished.emit(True,mv,ms)
        except Exception as e:
            import traceback; self.log.emit(f"[ERROR] {e}\n{traceback.format_exc()}"); self.finished.emit(False,"","")


class BatchPropagationThread(QThread):
    """Processes all annotated videos in sequence without user intervention."""
    video_started  = pyqtSignal(str, int, int)   # video_name, current_idx, total
    progress       = pyqtSignal(int, int)          # frames_done, total for current video
    log            = pyqtSignal(str)
    overlap_alert  = pyqtSignal(str, int, float)   # video_name, frame, iou
    video_done     = pyqtSignal(str, bool)          # video_name, success
    all_done       = pyqtSignal()

    def __init__(self, queue, out_dir):
        """queue: list of (video_path, start_frame, points) tuples in order."""
        super().__init__()
        self.queue = queue
        self.out_dir = out_dir
        self._cancel = False
        self._current_worker = None

    def cancel(self):
        self._cancel = True
        if self._current_worker:
            self._current_worker._cancel = True

    def run(self):
        total = len(self.queue)
        for idx, (video_path, start_frame, points) in enumerate(self.queue):
            if self._cancel: break
            vname = os.path.basename(video_path)
            self.video_started.emit(vname, idx+1, total)
            self.log.emit(f"\n{'─'*50}")
            self.log.emit(f"[{idx+1}/{total}] Starting: {vname}  (frame {start_frame})")
            self.log.emit(f"{'─'*50}")

            # Reuse PropagationThread logic inline so we share progress signals
            worker = PropagationThread(video_path, points, start_frame, self.out_dir)
            self._current_worker = worker

            # Wire worker signals to our own
            worker.progress.connect(self.progress)
            worker.log.connect(self.log)
            worker.overlap_alert.connect(lambda f,iou,vn=vname: self.overlap_alert.emit(vn,f,iou))

            # Run synchronously (we are already in a thread)
            worker.run()

            success = not worker._cancel
            if success:
                vid_out, shd_out = get_output_paths(video_path, self.out_dir, start_frame)
                add_segment(self.out_dir, video_path, start_frame, vid_out, shd_out)
                delete_annotation(self.out_dir, video_path)
            self.video_done.emit(vname, success)

        self._current_worker = None
        self.all_done.emit()

class ClickableFrame(QWidget):
    """Frame display with fluorescence-style point annotation overlay."""
    clicked_orig = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.original_frame = None
        self.points = []
        self._loading = False
        self.setStyleSheet(f"background:{BG};")

    def set_frame(self, frame_rgb, points=None):
        self.original_frame = frame_rgb
        if points is not None: self.points = points
        self._loading = False
        self.update()

    def set_points(self, points): self.points = points; self.update()
    def set_loading(self, v): self._loading = v; self.update()

    def _display_rect(self):
        if self.original_frame is None: return 0,0,0,0,1.0
        H,W = self.original_frame.shape[:2]
        lw,lh = self.width(), self.height()
        s = min(lw/W, lh/H)
        dw,dh = int(W*s), int(H*s)
        return (lw-dw)//2, (lh-dh)//2, dw, dh, s

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W,H = self.width(), self.height()

        # Background
        p.fillRect(0,0,W,H, QColor(BG))

        if self.original_frame is None:
            # Empty state
            p.setPen(QColor(TEXT_M)); p.setFont(QFont("Segoe UI",14))
            p.drawText(self.rect(), Qt.AlignCenter, "← Load a frame to begin annotation")
            return

        # Draw frame
        xo,yo,dw,dh,s = self._display_rect()
        fH,fW = self.original_frame.shape[:2]
        scaled = cv2.resize(self.original_frame,(dw,dh))
        img = QImage(scaled.data,dw,dh,dw*3,QImage.Format_RGB888).copy()
        p.drawImage(xo,yo,img)

        # Border glow around frame
        if self.points:
            pen = QPen(QColor(CYAN), 1.5)
            p.setPen(pen); p.setBrush(Qt.NoBrush)
            p.drawRect(xo, yo, dw, dh)

        # Draw annotation points
        for i,(oid,ox,oy) in enumerate(self.points):
            px = int(ox*s)+xo; py = int(oy*s)+yo
            color = OBJ_COLORS_QT.get(oid, QColor(255,255,255))

            # Glow ring
            glow = QRadialGradient(px, py, 16)
            gc = QColor(color); gc.setAlpha(50)
            glow.setColorAt(0, gc); glow.setColorAt(1, QColor(0,0,0,0))
            p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(px,py), 16, 16)

            # Solid circle
            p.setBrush(QBrush(color))
            p.setPen(QPen(Qt.white, 1.5))
            p.drawEllipse(QPointF(px,py), 8, 8)

            # Label
            p.setPen(QPen(Qt.white))
            p.setFont(QFont("Segoe UI",9,QFont.Bold))
            p.drawText(px+11, py-9, str(oid))

        # Loading overlay
        if self._loading:
            p.fillRect(xo,yo,dw,dh, QColor(0,0,0,140))
            p.setPen(QColor(CYAN)); p.setFont(QFont("Segoe UI",14))
            p.drawText(xo,yo,dw,dh, Qt.AlignCenter, "Computing preview…")

        p.end()

    def mousePressEvent(self, event):
        if self.original_frame is None: return
        xo,yo,dw,dh,s = self._display_rect()
        cx,cy = event.x(), event.y()
        fH,fW = self.original_frame.shape[:2]
        if xo<=cx<=xo+dw and yo<=cy<=yo+dh:
            self.clicked_orig.emit(
                max(0,min(fW-1, int((cx-xo)/s))),
                max(0,min(fH-1, int((cy-yo)/s)))
            )


# ═══════════════════════════ STYLED WIDGETS ════════════════════════════

def make_btn(text, obj_name="", icon=""):
    b = QPushButton(f"{icon}  {text}" if icon else text)
    if obj_name: b.setObjectName(obj_name)
    return b

def section_label(text):
    l = QLabel(text.upper())
    l.setStyleSheet(f"color:{TEXT_M}; font-size:10px; font-weight:bold; letter-spacing:2px; padding:8px 0 2px 0;")
    return l

def divider():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{BORDER}; max-height:1px;")
    return f


# ═══════════════════════════ MAIN WINDOW ═══════════════════════════════

QSS = f"""
* {{
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
    color: {TEXT};
    background: {BG};
}}

/* ── Header bar ── */
#header {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #020608, stop:0.5 #040d14, stop:1 #020608);
    border-bottom: 1px solid {BORDER};
    min-height: 46px; max-height: 46px;
}}
#header_title {{
    color: {CYAN};
    font-size: 17px; font-weight: bold; letter-spacing: 1px;
}}
#header_sub {{
    color: {TEXT_M}; font-size: 11px;
}}

/* ── Left panel ── */
#left_panel {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

/* ── Section labels ── */
#section_lbl {{
    color: {TEXT_M}; font-size: 10px; font-weight: bold;
    letter-spacing: 2px; padding: 10px 0 3px 0;
}}

/* ── Folder path labels ── */
#path_lbl {{
    color: {TEXT_S}; font-size: 11px; font-style: italic;
    padding: 2px 0;
}}

/* ── Video list ── */
QListWidget {{
    background: #060d12;
    border: 1px solid {BORDER};
    border-radius: 6px;
    outline: none;
    padding: 2px;
}}
QListWidget::item {{
    padding: 9px 12px;
    border-radius: 4px;
    color: {TEXT_S};
    border-bottom: 1px solid {PANEL};
}}
QListWidget::item:hover {{
    background: {SURFACE};
    color: {TEXT};
}}
QListWidget::item:selected {{
    background: #0a2030;
    color: {CYAN};
    border-left: 3px solid {CYAN};
    padding-left: 9px;
}}

/* ── Buttons ── */
QPushButton {{
    background: {SURFACE};
    color: {TEXT_S};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 7px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background: {SURFACE_H};
    color: {TEXT};
    border-color: {BORDER_A};
}}
QPushButton:pressed {{ background: {PANEL}; }}
QPushButton:disabled {{ color: {TEXT_M}; border-color: {BORDER}; }}

QPushButton#primary {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #006e82, stop:1 #004d5e);
    color: {CYAN_GLOW};
    border: 1px solid {CYAN};
    font-weight: bold;
}}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #008fa8, stop:1 #006e82);
    border-color: {CYAN_GLOW};
}}

QPushButton#success {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #005e36, stop:1 #003d22);
    color: {GREEN};
    border: 1px solid {GREEN};
    font-weight: bold;
}}
QPushButton#success:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #007a46, stop:1 #005e36);
}}

QPushButton#danger {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #6b0020, stop:1 #480016);
    color: {MAGENTA};
    border: 1px solid {MAGENTA};
    font-weight: bold;
}}
QPushButton#danger:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #8a0028, stop:1 #6b0020);
}}

QPushButton#flat {{
    background: transparent; border: none; color: {TEXT_S};
    padding: 6px 10px;
}}
QPushButton#flat:hover {{ color: {TEXT}; background: {SURFACE}; border-radius:4px; }}

/* ── Inputs ── */
QLineEdit, QSpinBox {{
    background: #060d12;
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: #004d5e;
}}
QLineEdit:focus, QSpinBox:focus {{ border-color: {CYAN}; }}
QSpinBox::up-button, QSpinBox::down-button {{
    background: {SURFACE}; border: none; width: 18px;
}}

/* ── Progress bar ── */
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    background: #060d12;
    height: 22px;
    text-align: center;
    color: {TEXT};
    font-weight: bold; font-size: 11px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {CYAN_DARK}, stop:0.5 {CYAN}, stop:1 {CYAN_DARK});
    border-radius: 4px;
}}

/* ── Log terminal ── */
#log_main {{
    background: #020508;
    border: 1px solid {BORDER};
    border-radius: 5px;
    font-family: 'Cascadia Code','Consolas','Courier New',monospace;
    font-size: 11px;
    color: {TERMINAL};
    selection-background-color: #1a3a50;
}}
#log_alert {{
    background: #0a0408;
    border: 1px solid #3a1020;
    border-radius: 5px;
    font-family: 'Cascadia Code','Consolas',monospace;
    font-size: 11px;
    color: {MAGENTA};
}}

/* ── Radio buttons ── */
QRadioButton {{ color: {TEXT}; spacing: 8px; padding: 4px 0; }}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border-radius: 7px;
    border: 2px solid {BORDER};
    background: #060d12;
}}
QRadioButton::indicator:checked {{ border-color: {CYAN}; background: {CYAN}; }}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: {PANEL}; width: 6px; border-radius: 3px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 3px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {CYAN_DARK}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 6px; background: {PANEL}; border-radius: 3px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 3px; }}

/* ── Status bar ── */
QStatusBar {{
    background: #020508;
    color: {TEXT_M};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 2px 8px;
}}

/* ── Splitter ── */
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QSplitter::handle:hover {{ background: {CYAN_DARK}; }}

/* ── Title labels ── */
#prop_title {{
    font-size: 15px; font-weight: bold; color: {CYAN};
    padding-bottom: 4px;
}}
#vid_lbl {{
    color: {TEXT_S}; font-size: 11px; font-style: italic;
}}
#batch_progress {{
    color: {TEXT_M}; font-size: 11px; padding: 4px 0;
}}
#debug_lbl {{
    color: {TEXT_S}; font-size: 11px;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MouseTrack")
        self.resize(1360, 820)
        self.video_dir = self.output_dir = self.current_video = ""
        self.selected_videos = []  # list of individually selected video file paths
        self.current_frame_rgb = None
        self.points = []; self.current_obj_id = 1
        self.prop_thread = self.prev_thread = self.merge_thread = None
        self.batch_prop_thread = None
        self.setStyleSheet(QSS)
        self._build_ui()

    def _build_ui(self):
        root_w = QWidget(); self.setCentralWidget(root_w)
        root_v = QVBoxLayout(root_w); root_v.setContentsMargins(0,0,0,0); root_v.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────
        hdr = QWidget(); hdr.setObjectName("header")
        hdrl = QHBoxLayout(hdr); hdrl.setContentsMargins(18,0,18,0)
        hdrl.setSpacing(0)
        icon = QLabel()
        _icon_pixmap = QPixmap()
        _icon_pixmap.loadFromData(base64.b64decode(MOUSE_ICON_B64))
        icon.setPixmap(_icon_pixmap.scaledToHeight(34, Qt.SmoothTransformation))
        icon.setContentsMargins(0,0,8,0)
        hdrl.addWidget(icon, 0, Qt.AlignVCenter)
        title = QLabel("MouseTrack"); title.setObjectName("header_title")
        hdrl.addWidget(title, 0, Qt.AlignVCenter)
        hdrl.addSpacing(16)
        self.vid_lbl = QLabel("No video loaded"); self.vid_lbl.setObjectName("header_sub")
        hdrl.addWidget(self.vid_lbl, 0, Qt.AlignVCenter); hdrl.addStretch()
        root_v.addWidget(hdr)

        # ── Body ────────────────────────────────────────────────────────
        body = QSplitter(Qt.Horizontal); root_v.addWidget(body, 1)

        # ── LEFT: Batch panel ──────────────────────────────────────────
        left = QWidget(); left.setObjectName("left_panel")
        left.setFixedWidth(260)
        lv = QVBoxLayout(left); lv.setContentsMargins(12,12,12,12); lv.setSpacing(4)

        lv.addWidget(section_label("Video Selection"))
        self.folder_lbl = QLabel("No videos selected"); self.folder_lbl.setObjectName("path_lbl")
        self.folder_lbl.setWordWrap(True); lv.addWidget(self.folder_lbl)
        btn_folder = make_btn("Select Videos"); btn_folder.clicked.connect(self.on_select_folder)
        lv.addWidget(btn_folder)

        lv.addWidget(divider()); lv.addWidget(section_label("Output Folder"))
        self.out_lbl = QLabel("(same as video folder)"); self.out_lbl.setObjectName("path_lbl")
        self.out_lbl.setWordWrap(True); lv.addWidget(self.out_lbl)
        btn_out = make_btn("Select output folder"); btn_out.clicked.connect(self.on_select_output)
        lv.addWidget(btn_out)

        lv.addWidget(divider()); lv.addWidget(section_label("Video List"))
        self.video_list = QListWidget()
        self.video_list.currentItemChanged.connect(self.on_video_selected)
        lv.addWidget(self.video_list, 1)

        self.batch_lbl = QLabel(""); self.batch_lbl.setObjectName("batch_progress")
        lv.addWidget(self.batch_lbl)

        lv.addWidget(divider())
        self.annotated_lbl = QLabel(""); self.annotated_lbl.setObjectName("batch_progress")
        lv.addWidget(self.annotated_lbl)
        self.btn_batch = make_btn("Start Batch Propagation", "success", "📍")
        self.btn_batch.setToolTip("Propagate all videos that have saved annotations, one after the other.")
        self.btn_batch.clicked.connect(self.on_start_batch); lv.addWidget(self.btn_batch)
        body.addWidget(left)

        # ── RIGHT: Workflow ────────────────────────────────────────────
        self.stack = QStackedWidget(); body.addWidget(self.stack)

        # Page 0: Welcome
        p0 = QWidget()
        p0v = QVBoxLayout(p0); p0v.setAlignment(Qt.AlignCenter); p0v.setSpacing(10)

        welcome_icon = QLabel()
        _welcome_pixmap = QPixmap()
        _welcome_pixmap.loadFromData(base64.b64decode(MOUSE_ICON_B64))
        welcome_icon.setPixmap(_welcome_pixmap.scaledToHeight(64, Qt.SmoothTransformation))
        welcome_icon.setAlignment(Qt.AlignCenter)
        p0v.addWidget(welcome_icon)

        welcome_title = QLabel("MouseTrack")
        welcome_title.setStyleSheet(f"color:{CYAN}; font-size:22px; font-weight:bold;")
        welcome_title.setAlignment(Qt.AlignCenter)
        p0v.addWidget(welcome_title)

        welcome_sub = QLabel("Select your videos and choose one from the list to begin.")
        welcome_sub.setStyleSheet(f"color:{TEXT_M}; font-size:15px;")
        welcome_sub.setAlignment(Qt.AlignCenter)
        p0v.addWidget(welcome_sub)

        self.stack.addWidget(p0)

        # Page 1: Annotation
        p1 = QWidget(); p1v = QVBoxLayout(p1); p1v.setContentsMargins(16,12,16,12); p1v.setSpacing(8)

        # Frame loader row
        fl = QHBoxLayout(); fl.setSpacing(8)
        fl.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox(); self.frame_spin.setRange(0,9999999); self.frame_spin.setFixedWidth(110)
        fl.addWidget(self.frame_spin)
        btn_load = make_btn("Load", "", "⊕"); btn_load.clicked.connect(self.on_load_frame)
        fl.addWidget(btn_load); fl.addStretch()
        p1v.addLayout(fl)

        # Clickable frame
        self.img_widget = ClickableFrame()
        self.img_widget.clicked_orig.connect(self.on_image_click)
        p1v.addWidget(self.img_widget, 1)

        # Bottom controls
        bot = QHBoxLayout(); bot.setSpacing(10)

        # Object IDs
        id_grp = QWidget()
        id_l = QHBoxLayout(id_grp); id_l.setContentsMargins(0,0,0,0); id_l.setSpacing(4)
        id_l.addWidget(QLabel("Object:"))
        self.id_group = QButtonGroup()
        for oid in [1,2,3,4]:
            rb = QRadioButton(f" {oid} ")
            c = OBJ_COLORS_QT[oid]
            rb.setStyleSheet(f"QRadioButton{{color:rgb({c.red()},{c.green()},{c.blue()});font-weight:bold;}}"
                             f"QRadioButton::indicator:checked{{background:rgb({c.red()},{c.green()},{c.blue()});border-color:rgb({c.red()},{c.green()},{c.blue()});}}")
            if oid==1: rb.setChecked(True)
            self.id_group.addButton(rb, oid); id_l.addWidget(rb)
        self.id_group.buttonClicked.connect(lambda b: setattr(self,'current_obj_id',self.id_group.id(b)))
        bot.addWidget(id_grp)

        btn_prev = make_btn("View current masks"); btn_prev.clicked.connect(self.on_preview_mask)
        bot.addWidget(btn_prev)

        btn_reset = make_btn("Reset masks"); btn_reset.clicked.connect(self.on_reset_points)
        bot.addWidget(btn_reset)

        self.debug_lbl = QLabel(""); self.debug_lbl.setObjectName("debug_lbl"); bot.addWidget(self.debug_lbl,1)

        btn_save_next = make_btn("Save and go to next video", "success")
        btn_save_next.setToolTip("Save annotation for this video and move to the next one.\n"
                                  "Run all propagations later with 'Start Batch Propagation'.")
        btn_save_next.clicked.connect(self.on_save_and_next); bot.addWidget(btn_save_next)

        btn_prop = make_btn("Propagate current video", "primary", "▶")
        btn_prop.setToolTip("Propagate only this video immediately.")
        btn_prop.clicked.connect(self.on_start_propagation); bot.addWidget(btn_prop)
        p1v.addLayout(bot)
        self.stack.addWidget(p1)

        # Page 2: Propagation & Merge
        p2 = QWidget(); p2v = QVBoxLayout(p2); p2v.setContentsMargins(16,12,16,12); p2v.setSpacing(8)
        self.prop_title = QLabel(); self.prop_title.setObjectName("prop_title"); p2v.addWidget(self.prop_title)
        self.prog_bar = QProgressBar(); self.prog_bar.setTextVisible(True); p2v.addWidget(self.prog_bar)

        self.prop_log = QPlainTextEdit(); self.prop_log.setObjectName("log_main"); self.prop_log.setReadOnly(True)
        p2v.addWidget(self.prop_log, 3)

        alert_hdr = QLabel("⚡  Identity-swap alerts (IoU > 0.3):")
        alert_hdr.setStyleSheet(f"color:{MAGENTA};font-weight:bold;font-size:11px;margin-top:4px;")
        p2v.addWidget(alert_hdr)
        self.alert_log = QPlainTextEdit(); self.alert_log.setObjectName("log_alert")
        self.alert_log.setReadOnly(True); self.alert_log.setMaximumHeight(90)
        p2v.addWidget(self.alert_log)

        btns2 = QHBoxLayout(); btns2.setSpacing(8)
        self.btn_cancel = make_btn("Cancel", "danger"); self.btn_cancel.clicked.connect(self.on_cancel)
        btns2.addWidget(self.btn_cancel)
        self.btn_back = make_btn("← Back"); self.btn_back.setObjectName("flat")
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_back.setVisible(False); btns2.addWidget(self.btn_back)
        btns2.addStretch()
        self.btn_merge = make_btn("Merge All Segments", "success", "⚙")
        self.btn_merge.clicked.connect(self.on_merge_segments); self.btn_merge.setVisible(False)
        btns2.addWidget(self.btn_merge)
        self.btn_next_video = make_btn("Next Video →", "primary")
        self.btn_next_video.clicked.connect(self.on_next_video); self.btn_next_video.setVisible(False)
        btns2.addWidget(self.btn_next_video)
        p2v.addLayout(btns2)
        self.stack.addWidget(p2)

        # Page 3: Batch propagation
        p3 = QWidget(); p3v = QVBoxLayout(p3); p3v.setContentsMargins(16,12,16,12); p3v.setSpacing(8)
        self.batch_title = QLabel(); self.batch_title.setObjectName("prop_title"); p3v.addWidget(self.batch_title)

        # Overall batch progress
        overall_row = QHBoxLayout()
        overall_row.addWidget(QLabel("Overall:"))
        self.batch_overall_bar = QProgressBar(); self.batch_overall_bar.setTextVisible(True)
        overall_row.addWidget(self.batch_overall_bar)
        p3v.addLayout(overall_row)

        # Current video progress
        current_row = QHBoxLayout()
        self.batch_current_lbl = QLabel("Current:"); current_row.addWidget(self.batch_current_lbl)
        self.batch_video_bar = QProgressBar(); self.batch_video_bar.setTextVisible(True)
        current_row.addWidget(self.batch_video_bar)
        p3v.addLayout(current_row)

        self.batch_log = QPlainTextEdit(); self.batch_log.setObjectName("log_main"); self.batch_log.setReadOnly(True)
        p3v.addWidget(self.batch_log, 3)

        batch_alert_hdr = QLabel("⚡  Identity-swap alerts:")
        batch_alert_hdr.setStyleSheet(f"color:{MAGENTA};font-weight:bold;font-size:11px;margin-top:4px;")
        p3v.addWidget(batch_alert_hdr)
        self.batch_alert_log = QPlainTextEdit(); self.batch_alert_log.setObjectName("log_alert")
        self.batch_alert_log.setReadOnly(True); self.batch_alert_log.setMaximumHeight(80)
        p3v.addWidget(self.batch_alert_log)

        btns3 = QHBoxLayout()
        self.btn_batch_cancel = make_btn("Cancel Batch", "danger")
        self.btn_batch_cancel.clicked.connect(self.on_cancel_batch); btns3.addWidget(self.btn_batch_cancel)
        btns3.addStretch()
        self.btn_batch_done = make_btn("Back to Annotation", "flat")
        self.btn_batch_done.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_batch_done.setVisible(False); btns3.addWidget(self.btn_batch_done)
        p3v.addLayout(btns3)
        self.stack.addWidget(p3)

        self.statusBar().showMessage("Ready")

    # ── Folder / video selection ───────────────────────────────────────
    def _get_video_list(self):
        """Returns the currently selected list of video file paths."""
        return sorted(self.selected_videos)

    def on_select_folder(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Videos (" + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + ");;All files (*)"
        )
        if not files: return
        # Accumulate: add newly picked files to whatever was already
        # selected, instead of replacing the list. Avoid duplicates
        # (same file picked twice) while preserving order.
        existing = set(self.selected_videos)
        for f in files:
            if f not in existing:
                self.selected_videos.append(f)
                existing.add(f)
        if not self.video_dir:
            self.video_dir = os.path.dirname(files[0])  # used as default location for output dir
        n = len(self.selected_videos)
        self.folder_lbl.setText(f"{n} video(s) selected")
        if not self.output_dir:
            self.output_dir = os.path.join(self.video_dir,"sam3_output"); self.out_lbl.setText("sam3_output/")
        self._refresh()

    def on_select_output(self):
        d = QFileDialog.getExistingDirectory(self,"Select Output Folder")
        if d: self.output_dir=d; self.out_lbl.setText(os.path.basename(d)); self._refresh()

    def _refresh(self):
        self.video_list.clear()
        videos = self._get_video_list()
        if not videos: return
        status = load_batch_status(self.output_dir) if self.output_dir else {}
        ann = load_annotations(self.output_dir) if self.output_dir else {}
        done=todo=annotated=0
        for v in videos:
            bn=os.path.basename(v); s=status.get(bn,{}).get("status","pending")
            has_ann = bn in ann
            # NOTE: has_ann must be checked FIRST. A video can already be
            # "done" from a previous full run, but if the user just saved a
            # new annotation for a resume/correction, that pending action
            # must be visible — otherwise it looks like nothing was saved.
            if has_ann:
                icon="📌"; annotated+=1  # annotated but not yet propagated
            elif s in ("done","merged"):
                icon="✅"; done+=1
            elif s=="has_resume":
                icon="⚠️"; todo+=1
            else:
                icon="○"; todo+=1
            item=QListWidgetItem(f"{icon}  {bn}"); item.setData(Qt.UserRole,v)
            if has_ann: item.setForeground(QColor(YELLOW))
            elif s in ("done","merged"): item.setForeground(QColor(TEXT_M))
            self.video_list.addItem(item)
        self.batch_lbl.setText(f"✅ {done} done")
        if annotated:
            self.annotated_lbl.setText(f"📌 {annotated} video(s) ready to propagate")
            self.btn_batch.setEnabled(True)
        else:
            self.annotated_lbl.setText("No saved annotations yet")
            self.btn_batch.setEnabled(False)

    def on_video_selected(self, item):
        if not item: return
        self.current_video=item.data(Qt.UserRole)
        self.vid_lbl.setText(f"  /  {os.path.basename(self.current_video)}")
        self.frame_spin.setValue(0); self.on_load_frame(); self.stack.setCurrentIndex(1)

    def on_next_video(self):
        if not self.selected_videos: QMessageBox.warning(self,"","Select videos first."); return
        status=load_batch_status(self.output_dir) if self.output_dir else {}
        videos=self._get_video_list()
        for v in videos:
            if status.get(os.path.basename(v),{}).get("status","pending") not in ("done","merged"):
                self.current_video=v; self.vid_lbl.setText(f"  /  {os.path.basename(v)}")
                self.frame_spin.setValue(0); self.on_load_frame(); self.stack.setCurrentIndex(1)
                for i in range(self.video_list.count()):
                    if self.video_list.item(i).data(Qt.UserRole)==v:
                        self.video_list.setCurrentRow(i); break
                return
        QMessageBox.information(self,"","All videos are done! 🎉")

    # ── Frame & annotation ────────────────────────────────────────────
    def on_load_frame(self):
        if not self.current_video: return
        fi=self.frame_spin.value()
        cap=cv2.VideoCapture(self.current_video)
        self.frame_spin.setMaximum(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        if fi>0: cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        ok,f=cap.read(); cap.release()
        if not ok: QMessageBox.warning(self,"Error",f"Cannot read frame {fi}."); return
        self.current_frame_rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        self.points=[]; self.img_widget.set_frame(self.current_frame_rgb,[])
        self.debug_lbl.setText(f"Frame {fi} — click on subjects.")
        self.statusBar().showMessage(f"Frame {fi} loaded.")

    def on_image_click(self,x,y):
        self.points.append((self.current_obj_id,x,y))
        self.img_widget.set_points(self.points)
        self.debug_lbl.setText(f"{len(self.points)} point(s) placed.")

    def on_reset_points(self):
        self.points=[]; self.img_widget.set_points([])
        if self.current_frame_rgb is not None: self.img_widget.set_frame(self.current_frame_rgb,[])
        self.debug_lbl.setText("Points reset.")

    def on_preview_mask(self):
        if not self.points: QMessageBox.information(self,"","Click on a subject first."); return
        self.img_widget.set_loading(True); self.debug_lbl.setText("Computing preview mask…")
        self.prev_thread=PreviewThread(self.current_frame_rgb,self.points)
        self.prev_thread.done.connect(self._preview_done); self.prev_thread.start()

    def _preview_done(self,img,txt):
        self.debug_lbl.setText(txt)
        if img is not None: self.img_widget.set_frame(img,self.points)
        else: self.img_widget.set_loading(False)

    # ── Propagation ──────────────────────────────────────────────────
    def on_start_propagation(self):
        if not self.current_video: QMessageBox.warning(self,"","Select a video first."); return
        if not self.points: QMessageBox.warning(self,"","Click on at least one subject."); return
        if not self.output_dir: QMessageBox.warning(self,"","Select an output folder first."); return
        sf=self.frame_spin.value(); self.stack.setCurrentIndex(2)
        self.prop_title.setText("⬡  Propagation in progress…")
        self.prop_log.clear(); self.alert_log.clear(); self.prog_bar.setValue(0)
        self.btn_cancel.setVisible(True); self.btn_cancel.setEnabled(True)
        self.btn_back.setVisible(False); self.btn_merge.setVisible(False); self.btn_next_video.setVisible(False)
        self.prop_thread=PropagationThread(self.current_video,self.points,sf,self.output_dir)
        self.prop_thread.progress.connect(lambda d,t: (self.prog_bar.setValue(int(d/t*100)),
            self.prog_bar.setFormat(f"{d}/{t}  ({int(d/t*100)}%)")))
        self.prop_thread.log.connect(self.prop_log.appendPlainText)
        self.prop_thread.overlap_alert.connect(
            lambda f,iou: self.alert_log.appendPlainText(f"Frame {f}: IoU={iou:.2f}"))
        self.prop_thread.finished.connect(self._prop_done); self.prop_thread.start()

    def on_save_and_next(self):
        """Save current annotation to disk and move to the next unannotated video."""
        if not self.current_video:
            QMessageBox.warning(self,"","Select a video first."); return
        if not self.points:
            QMessageBox.warning(self,"","Click on at least one subject first."); return
        if not self.output_dir:
            QMessageBox.warning(self,"","Select an output folder first."); return
        sf = self.frame_spin.value()
        save_annotation(self.output_dir, self.current_video, sf, self.points)
        self._refresh()
        self.statusBar().showMessage(f"Annotation saved for {os.path.basename(self.current_video)}.")
        # Move to next unannotated/unprocessed video
        ann = load_annotations(self.output_dir)
        status = load_batch_status(self.output_dir)
        videos = self._get_video_list()
        for v in videos:
            bn = os.path.basename(v)
            s = status.get(bn,{}).get("status","pending")
            if s not in ("done","merged") and bn not in ann and v != self.current_video:
                self.current_video = v
                self.vid_lbl.setText(f"  /  {bn}")
                self.frame_spin.setValue(0); self.on_load_frame()
                for i in range(self.video_list.count()):
                    if self.video_list.item(i).data(Qt.UserRole)==v:
                        self.video_list.setCurrentRow(i); break
                return
        self.debug_lbl.setText("All videos annotated — ready for batch propagation!")

    def on_start_batch(self):
        if not self.output_dir: QMessageBox.warning(self,"","Select an output folder first."); return
        ann = load_annotations(self.output_dir)
        if not ann: QMessageBox.information(self,"","No saved annotations found.\nUse 'Save and go to next video' to annotate videos first."); return
        # Build queue: only videos that exist on disk among the selected ones
        video_by_name = {os.path.basename(v): v for v in self.selected_videos}
        queue = []
        for bn, entry in ann.items():
            vpath = video_by_name.get(bn)
            if vpath and os.path.exists(vpath):
                queue.append((vpath, entry["start_frame"], entry["points"]))
        if not queue: QMessageBox.warning(self,"","No matching video files found for the saved annotations."); return

        total = len(queue)
        self.stack.setCurrentIndex(3)
        self.batch_title.setText(f"⬡  Batch propagation — {total} video(s) in queue")
        self.batch_log.clear(); self.batch_alert_log.clear()
        self.batch_overall_bar.setRange(0, total); self.batch_overall_bar.setValue(0)
        self.batch_overall_bar.setFormat(f"0 / {total} videos")
        self.batch_video_bar.setValue(0); self.batch_video_bar.setFormat("–")
        self.btn_batch_cancel.setEnabled(True); self.btn_batch_done.setVisible(False)

        self.batch_prop_thread = BatchPropagationThread(queue, self.output_dir)
        self.batch_prop_thread.video_started.connect(self._batch_video_started)
        self.batch_prop_thread.progress.connect(self._batch_frame_progress)
        self.batch_prop_thread.log.connect(self.batch_log.appendPlainText)
        self.batch_prop_thread.overlap_alert.connect(
            lambda vn,f,iou: self.batch_alert_log.appendPlainText(f"[{vn}] frame {f}: IoU={iou:.2f}"))
        self.batch_prop_thread.video_done.connect(self._batch_video_done)
        self.batch_prop_thread.all_done.connect(self._batch_all_done)
        self.batch_prop_thread.start()

    def _batch_video_started(self, vname, idx, total):
        self.batch_current_lbl.setText(f"Now: {vname}")
        self.batch_video_bar.setValue(0); self.batch_video_bar.setFormat("0%")
        self.batch_overall_bar.setFormat(f"{idx-1} / {total} videos done")
        self.statusBar().showMessage(f"[{idx}/{total}] Propagating {vname}…")

    def _batch_frame_progress(self, done, total):
        pct = int(done/total*100) if total else 0
        self.batch_video_bar.setValue(pct); self.batch_video_bar.setFormat(f"{done}/{total}  ({pct}%)")

    def _batch_video_done(self, vname, success):
        icon = "✅" if success else "✗"
        self.batch_log.appendPlainText(f"\n{icon} {vname} {'complete' if success else 'FAILED'}\n")
        done = self.batch_overall_bar.value() + (1 if success else 0)
        self.batch_overall_bar.setValue(done)
        self._refresh()

    def _batch_all_done(self):
        total = self.batch_overall_bar.maximum()
        done  = self.batch_overall_bar.value()
        self.batch_title.setText(f"✅  Batch complete — {done}/{total} videos processed.")
        self.batch_overall_bar.setFormat(f"{done} / {total} videos done")
        self.btn_batch_cancel.setEnabled(False); self.btn_batch_done.setVisible(True)
        self.statusBar().showMessage(f"Batch done: {done}/{total} videos.")
        self._refresh()

    def on_cancel(self):
        if self.prop_thread: self.prop_thread.cancel()
        self.btn_cancel.setEnabled(False)

    def on_cancel_batch(self):
        if self.batch_prop_thread: self.batch_prop_thread.cancel()
        self.batch_title.setText("⚠  Batch cancelled.")
        if self.batch_prop_thread: self.batch_prop_thread.cancel()
        self.batch_title.setText("⚠  Batch cancelled.")

    def _prop_done(self,vid,shd,ok):
        self.btn_cancel.setEnabled(False); self.btn_back.setVisible(True)
        if not ok: self.prop_title.setText("✗  Propagation cancelled."); return
        sf=self.frame_spin.value(); add_segment(self.output_dir,self.current_video,sf,vid,shd)
        self._refresh()
        segs=load_batch_status(self.output_dir).get(os.path.basename(self.current_video),{}).get("segments",[])
        if len(segs)>1:
            self.prop_title.setText("⚠  Segment saved — multiple segments detected.")
            self.btn_merge.setVisible(True)
        else:
            self.prop_title.setText("✅  Propagation complete!")
            self.btn_next_video.setVisible(True)
        self.prop_log.appendPlainText(f"\n── Output ──\nVideo:  {vid}\nShards: {shd}")
        self.statusBar().showMessage("Done.")

    # ── Merge ────────────────────────────────────────────────────────
    def on_merge_segments(self):
        segs=load_batch_status(self.output_dir).get(os.path.basename(self.current_video),{}).get("segments",[])
        if len(segs)<2: QMessageBox.information(self,"","Only one segment — nothing to merge."); return
        self.btn_merge.setEnabled(False); self.prop_title.setText("⚙  Merging all segments…")
        self.merge_thread=MergeThread(self.current_video,segs,self.output_dir)
        self.merge_thread.log.connect(self.prop_log.appendPlainText)
        self.merge_thread.finished.connect(self._merge_done); self.merge_thread.start()

    def _merge_done(self,ok,mv,ms):
        if ok:
            mark_merged(self.output_dir,self.current_video,mv,ms); self._refresh()
            self.prop_title.setText("✅  Merge complete! All segments unified.")
            self.prop_log.appendPlainText(f"\n── Merged ──\nVideo:  {mv}\nShards: {ms}")
            self.btn_next_video.setVisible(True)
        else:
            self.prop_title.setText("✗  Merge failed. Check the log.")
        self.btn_merge.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("SAM3 Tracker")
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
