"""
TrackFix_CD1
---------------------
GUI for correcting trajectory ID swaps specifically for CD1-BL6 pairs.

Unlike the generic TrackFix tool (which needs a separate centroid .npy file
and a distance threshold to flag "ambiguous" frames), this version detects
likely ID swaps directly from the video: CD1 mice are white, BL6 mice are
black. For every frame, it samples the pixel brightness under the 'center'
keypoint of each individual. If the individual labeled 'CD1' sits on a
darker patch than the individual labeled 'BL6', the two identities are very
likely swapped for that frame — and are flagged accordingly. Applying a fix
swaps ALL bodyparts for that frame (not just 'center'), since identity is a
per-individual, per-frame property in the tracking file.

Usage:
    python TrackFix_CD1.py [--patch_radius 4] [--color_margin 25] [--min_zone_len 2] [--likelihood_threshold 0.5]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QLabel, QListWidget, QListWidgetItem,
    QSpinBox, QGroupBox, QMessageBox, QSplitter, QStatusBar,
    QFileDialog, QSizePolicy, QFrame, QSplashScreen
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPainter, QColor, QPen, QBrush, QPainterPath

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--patch_radius",         type=int,   default=4,
                     help="Half-size (px) of the square patch sampled around the 'center' keypoint for brightness measurement.")
parser.add_argument("--color_margin",         type=float, default=25.0,
                     help="Minimum brightness difference (0-255 grayscale) required to flag a frame as swapped.")
parser.add_argument("--min_zone_len",         type=int,   default=2,
                     help="Minimum number of consecutive flagged frames to count as a suspected-swap zone (filters single-frame noise).")
parser.add_argument("--likelihood_threshold", type=float, default=0.5,
                     help="DLC likelihood below which a 'center' keypoint is ignored for color analysis.")
args = parser.parse_args()

# ── Identity constants ─────────────────────────────────────────────────────────
INDIVIDUAL_CD1 = 'CD1'   # white mouse
INDIVIDUAL_BL6 = 'BL6'   # black mouse

# ── Asset paths ────────────────────────────────────────────────────────────────
ASSETS_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SPLASH_IMAGE_PATH = os.path.join(ASSETS_DIR, "trackfix_splash.png")  # full splash (mouse + logo + text)
MOUSE_ICON_PATH   = os.path.join(ASSETS_DIR, "mouse_only.png")       # cropped mouse only, used as small icon

SPLASH_DURATION_MS = 2500  # how long the splash screen stays visible on launch


# ── App icon: mouse cropped from the brand image, falls back to a drawn icon ──
def make_mouse_icon():
    """Load the cropped mouse image to use as the app/window icon.
    Falls back to a small drawn placeholder if the asset is missing."""
    if os.path.isfile(MOUSE_ICON_PATH):
        px = QPixmap(MOUSE_ICON_PATH)
        if not px.isNull():
            return QIcon(px)

    # Fallback: simple drawn placeholder (used only if the asset can't be found)
    size = 64
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    cx, cy = size // 2, size // 2 + 4
    p.setBrush(QBrush(QColor("#cccccc")))
    p.setPen(QPen(QColor("#888888"), 1))
    p.drawEllipse(cx - 10, cy - 14, 20, 26)
    p.drawEllipse(cx - 8, cy - 24, 16, 16)
    p.setBrush(QBrush(QColor("#3399ff")))
    p.setPen(QPen(QColor("#1a66cc"), 1))
    p.drawEllipse(cx - 6, cy - 5, 12, 12)
    p.end()
    return QIcon(px)


def make_mouse_pixmap(size_px):
    """Return a square QPixmap of the cropped mouse image at the given size,
    for use inline next to the 'TrackFix' title label."""
    if os.path.isfile(MOUSE_ICON_PATH):
        px = QPixmap(MOUSE_ICON_PATH)
        if not px.isNull():
            return px.scaled(size_px, size_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return make_mouse_icon().pixmap(QSize(size_px, size_px))


# ── Constants ──────────────────────────────────────────────────────────────────
BODYPARTS = ['nose','right_ear','left_ear','right_fhip','left_fhip',
             'spine_1','center','right_bhip','left_bhip','spine_2',
             'tail_base','tail_1','tail_2','tail_tip']

SKELETON = [
    ('nose','right_ear'), ('nose','left_ear'),
    ('right_ear','spine_1'), ('left_ear','spine_1'),
    ('spine_1','center'), ('center','spine_2'),
    ('spine_2','right_bhip'), ('spine_2','left_bhip'),
    ('right_fhip','spine_1'), ('left_fhip','spine_1'),
    ('spine_2','tail_base'), ('tail_base','tail_1'),
    ('tail_1','tail_2'), ('tail_2','tail_tip')
]

# Annotation colors for drawing the skeletons — chosen purely for visibility
# on screen, independent of the mice's actual fur color. The actual white/black
# fur color is only used internally, for the brightness-based swap detection.
COLOR_CD1      = (139, 0, 0)      # CD1 (white mouse) skeleton — drawn in dark blue (BGR: B=139)
COLOR_BL6      = (0, 0, 200)     # BL6 (black mouse) skeleton — drawn in red (BGR: R=200)
COLOR_FLAGGED  = (0, 200, 255)   # border/highlight for suspected-swap frames
COLOR_SAMPLE_CD1 = (255, 255, 255)  # sample-patch outline for CD1 (white)
COLOR_SAMPLE_BL6 = (255, 140, 0)    # sample-patch outline for BL6 (orange; true black wouldn't be visible)

# ── Dark theme stylesheet ──────────────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 10px;
}
QPushButton:hover {
    background-color: #45475a;
    border: 1px solid #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border: 1px solid #313244;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #45475a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 2px;
}
QListWidget {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
}
QListWidget::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QListWidget::item:hover {
    background-color: #313244;
}
QSpinBox {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
    padding: 2px 4px;
}
QSplitter::handle {
    background-color: #45475a;
    width: 2px;
}
QStatusBar {
    background-color: #181825;
    color: #6c7086;
    border-top: 1px solid #45475a;
    font-size: 11px;
}
QLabel {
    color: #cdd6f4;
}
"""

# ── Global state ───────────────────────────────────────────────────────────────
state = {
    'df':             None,
    'scorer':         None,
    'cd1_map':        None,   # {bodypart: (x_idx, y_idx, likelihood_idx)} for CD1
    'bl6_map':        None,   # same for BL6
    'cd1_idx':        None,   # full column-index list for CD1 (all bodyparts) — used for swapping
    'bl6_idx':        None,   # full column-index list for BL6
    'brightness_cd1': np.array([]),
    'brightness_bl6': np.array([]),
    'color_flagged':  np.array([], dtype=bool),
    'zones':          [],     # suspected-swap zones, derived from color_flagged
    'n_frames':       0,
    'dlc_path':       None,
    'out_path':       None,
}

def col_idx_map(df, scorer, individual):
    """Map each bodypart to (x_idx, y_idx, likelihood_idx) column positions,
    matching bodypart names case-insensitively against this CSV's header."""
    m = {}
    actual_bp_names = {}
    for c in df.columns:
        if c[1] == individual:
            actual_bp_names[c[2].lower()] = c[2]
    for bp in BODYPARTS:
        actual_bp = actual_bp_names.get(bp.lower())
        if actual_bp is None:
            continue
        try:
            m[bp] = (df.columns.get_loc((scorer, individual, actual_bp, 'x')),
                      df.columns.get_loc((scorer, individual, actual_bp, 'y')),
                      df.columns.get_loc((scorer, individual, actual_bp, 'likelihood')))
        except:
            pass
    return m

def load_dlc(path):
    df = pd.read_csv(path, header=[0,1,2,3], index_col=0, low_memory=False)
    scorer = df.columns[0][0]
    state['df']       = df.copy()
    state['scorer']   = scorer
    state['dlc_path'] = path
    state['out_path'] = path.replace('.csv', '_corrected.csv')
    _update_maps()
    # Suspected-swap zones require sampling the video, so they're computed
    # once both the CSV and the video are loaded (see MainWindow._maybe_run_color_analysis).
    state['zones']         = []
    state['brightness_cd1'] = np.array([])
    state['brightness_bl6'] = np.array([])
    state['color_flagged']  = np.array([], dtype=bool)

def _check_individuals(df):
    """Verify the CSV header contains exactly CD1 and BL6 as individuals."""
    seen = []
    for c in df.columns:
        name = c[1]
        if name not in seen:
            seen.append(name)
    if set(seen) != {INDIVIDUAL_CD1, INDIVIDUAL_BL6}:
        raise ValueError(
            f"This tool expects exactly the two individuals '{INDIVIDUAL_CD1}' and "
            f"'{INDIVIDUAL_BL6}' in the CSV header, but found: {seen}. "
            f"Use the generic TrackFix tool for other individual-name conventions."
        )

def _update_maps():
    df     = state['df']
    scorer = state['scorer']
    if df is None:
        return
    _check_individuals(df)
    state['cd1_map'] = col_idx_map(df, scorer, INDIVIDUAL_CD1)
    state['bl6_map'] = col_idx_map(df, scorer, INDIVIDUAL_BL6)
    cd1_cols = [c for c in df.columns if c[1] == INDIVIDUAL_CD1]
    bl6_cols = [c for c in df.columns if c[1] == INDIVIDUAL_BL6]
    state['cd1_idx'] = [df.columns.get_loc(c) for c in cd1_cols]
    state['bl6_idx'] = [df.columns.get_loc(c) for c in bl6_cols]
    state['n_frames'] = len(df)

def get_keypoints(frame, cmap):
    df = state['df']
    kps = {}
    for bp in BODYPARTS:
        if bp not in cmap:
            continue
        xi, yi, _li = cmap[bp]
        try:
            x, y = float(df.iloc[frame, xi]), float(df.iloc[frame, yi])
            if not (np.isnan(x) or np.isnan(y)):
                kps[bp] = (int(x), int(y))
        except:
            pass
    return kps

def get_center_xy_likelihood(frame, cmap):
    """Return (x, y, likelihood) for the 'center' keypoint, or (None, None, None)
    if unavailable/NaN."""
    if 'center' not in cmap:
        return None, None, None
    xi, yi, li = cmap['center']
    df = state['df']
    try:
        x, y, l = float(df.iloc[frame, xi]), float(df.iloc[frame, yi]), float(df.iloc[frame, li])
        if np.isnan(x) or np.isnan(y):
            return None, None, None
        return x, y, (l if not np.isnan(l) else None)
    except Exception:
        return None, None, None

def sample_patch_brightness(gray_img, x, y, radius):
    """Mean grayscale intensity (0-255) of a square patch centered on (x, y)."""
    if x is None or y is None:
        return None
    h, w = gray_img.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = gray_img[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return float(patch.mean())

def analyze_color_swaps(cap, progress_cb=None):
    """Walk through every frame of the video, sample brightness under each
    individual's 'center' keypoint, and flag frames where 'CD1' sits on a
    darker patch than 'BL6' by more than args.color_margin — i.e. frames
    where the identities are very likely swapped."""
    df = state['df']
    n = min(len(df), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cd1_map = state['cd1_map']
    bl6_map = state['bl6_map']

    brightness_cd1 = np.full(n, np.nan)
    brightness_bl6 = np.full(n, np.nan)
    flagged        = np.zeros(n, dtype=bool)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(n):
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x1, y1, l1 = get_center_xy_likelihood(f, cd1_map)
        x2, y2, l2 = get_center_xy_likelihood(f, bl6_map)
        if l1 is not None and l1 < args.likelihood_threshold:
            x1, y1 = None, None
        if l2 is not None and l2 < args.likelihood_threshold:
            x2, y2 = None, None

        b1 = sample_patch_brightness(gray, x1, y1, args.patch_radius)
        b2 = sample_patch_brightness(gray, x2, y2, args.patch_radius)
        brightness_cd1[f] = b1 if b1 is not None else np.nan
        brightness_bl6[f] = b2 if b2 is not None else np.nan

        # Expect CD1 (white) brighter than BL6 (black). If BL6 is brighter
        # than CD1 by more than the margin, the labels are likely swapped.
        if b1 is not None and b2 is not None and (b2 - b1) > args.color_margin:
            flagged[f] = True

        if progress_cb is not None and f % 50 == 0:
            progress_cb(f, n)

    state['brightness_cd1'] = brightness_cd1
    state['brightness_bl6'] = brightness_bl6
    state['color_flagged']  = flagged
    state['n_frames']       = n

    zones = []
    i = 0
    while i < n:
        if flagged[i]:
            j = i
            while j < n and flagged[j]:
                j += 1
            if (j - i) >= args.min_zone_len:
                zones.append((i, j))
            i = j
        else:
            i += 1
    state['zones'] = zones

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if progress_cb is not None:
        progress_cb(n, n)

def draw_skeleton(img, kps, color):
    for (bp1, bp2) in SKELETON:
        if bp1 in kps and bp2 in kps:
            cv2.line(img, kps[bp1], kps[bp2], color, 2)
    for bp, pos in kps.items():
        cv2.circle(img, pos, 4, color, -1)

def draw_sample_patch(img, x, y, radius, color):
    if x is None or y is None:
        return
    xi, yi = int(round(x)), int(round(y))
    cv2.rectangle(img, (xi - radius, yi - radius), (xi + radius, yi + radius), color, 2)

def apply_swap(frame):
    """Swap ALL bodyparts between CD1 and BL6 for this frame — identity is
    swapped as a whole, not just the 'center' keypoint used for detection."""
    df = state['df']
    ci = state['cd1_idx']
    bi = state['bl6_idx']
    cv_ = df.iloc[frame, ci].values.copy()
    bv  = df.iloc[frame, bi].values.copy()
    df.iloc[frame, ci] = bv
    df.iloc[frame, bi] = cv_

def apply_swap_range(start, end):
    for f in range(start, end + 1):
        apply_swap(f)

def undo_swap_range(start, end):
    for f in range(start, end + 1):
        apply_swap(f)

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_btn(text, color=None, bold=False):
    btn = QPushButton(text)
    style = ""
    if color:
        style += f"background-color: {color}; color: white;"
    if bold:
        style += "font-weight: bold;"
    style += "border-radius: 5px; padding: 5px 10px;"
    btn.setStyleSheet(style)
    return btn

def make_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #45475a;")
    return line


# ── Splash screen ──────────────────────────────────────────────────────────────
def make_splash_pixmap():
    """Build the splash pixmap: the full brand image (mouse + 'TrackFix' + subtext)
    centered on a black background."""
    if os.path.isfile(SPLASH_IMAGE_PATH):
        pix = QPixmap(SPLASH_IMAGE_PATH)
        if not pix.isNull():
            # Scale down a bit so it isn't oversized on screen, keep aspect ratio
            target = 560
            pix = pix.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix

    # Fallback: plain black pixmap with text, in case the asset is missing
    pix = QPixmap(560, 560)
    pix.fill(QColor("#000000"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor("#cdd6f4"))
    p.setFont(QFont("Segoe UI", 28, QFont.Bold))
    p.drawText(pix.rect(), Qt.AlignCenter, "TrackFix")
    p.end()
    return pix


class TrackFixSplashScreen(QSplashScreen):
    """Splash screen shown on launch: brand image on a black background."""
    def __init__(self):
        pixmap = make_splash_pixmap()
        # Pad the pixmap onto a pure black square canvas so the background is solid black
        canvas = QPixmap(pixmap.width(), pixmap.height())
        canvas.fill(QColor("#000000"))
        p = QPainter(canvas)
        p.drawPixmap(0, 0, pixmap)
        p.end()
        super().__init__(canvas)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)


# ── Main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackFix")
        self.setWindowIcon(make_mouse_icon())
        self.setMinimumSize(1400, 850)

        self.cap           = None
        self.current_frame = 0
        self.swap_ranges   = []
        # Position of the frame most recently returned by self.cap.read(), so
        # sequential steps can just read the next frame (which OpenCV decodes
        # frame-accurately) instead of re-seeking every time. -1 = unknown.
        self._cap_pos      = -1

        # Slider scrubbing throttle: while the scrollbar is dragged, valueChanged
        # fires for every intermediate value. Rendering each one floods the event
        # queue and the display lags behind the cursor. Instead we remember the
        # latest requested frame and render on a short timer, dropping the frames
        # in between so scrubbing stays responsive.
        self._pending_slider_frame = None
        self._slider_render_timer  = QTimer(self)
        self._slider_render_timer.setSingleShot(True)
        self._slider_render_timer.setInterval(25)  # ms between scrub renders (~40/s max)
        self._slider_render_timer.timeout.connect(self._render_pending_slider_frame)

        self._build_ui()
        self._show_placeholder()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left: video ────────────────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Title bar — small mouse logo (cropped from the brand image) + "TrackFix"
        title_layout = QHBoxLayout()
        title_layout.setSpacing(8)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(make_mouse_pixmap(32))
        title_layout.addWidget(icon_lbl)
        title_lbl = QLabel("TrackFix")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; letter-spacing: 2px;")
        title_layout.addWidget(title_lbl)
        title_layout.addStretch()
        left_layout.addLayout(title_layout)

        left_layout.addWidget(make_separator())

        # File buttons
        file_layout = QHBoxLayout()
        file_layout.setSpacing(6)
        self.btn_open_csv = make_btn("📄  Open Tracking csv file")
        self.btn_open_csv.setFixedHeight(32)
        self.btn_open_csv.clicked.connect(self._open_csv)
        file_layout.addWidget(self.btn_open_csv)

        self.btn_open_video = make_btn("🎬  Open original video")
        self.btn_open_video.setFixedHeight(32)
        self.btn_open_video.clicked.connect(self._open_video)
        file_layout.addWidget(self.btn_open_video)
        left_layout.addLayout(file_layout)

        # Status labels in one row
        status_layout = QHBoxLayout()
        status_layout.setSpacing(16)
        self.lbl_csv   = QLabel("CSV: —")
        self.lbl_video = QLabel("Video: —")
        for lbl in [self.lbl_csv, self.lbl_video]:
            lbl.setStyleSheet("color: #585b70; font-size: 10px;")
            lbl.setFixedHeight(14)
            status_layout.addWidget(lbl)
        status_layout.addStretch()
        left_layout.addLayout(status_layout)

        left_layout.addWidget(make_separator())

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #11111b; border-radius: 6px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(400, 300)
        left_layout.addWidget(self.video_label, stretch=1)

        # Frame info
        info_layout = QHBoxLayout()
        self.frame_label = QLabel("No data loaded")
        self.frame_label.setStyleSheet("color: #6c7086; font-size: 11px; font-family: monospace;")
        self.frame_label.setFixedHeight(18)
        self.zone_label = QLabel("")
        self.zone_label.setStyleSheet("color: #fab387; font-weight: bold; font-size: 11px;")
        self.zone_label.setFixedHeight(18)
        info_layout.addWidget(self.frame_label)
        info_layout.addStretch()
        info_layout.addWidget(self.zone_label)
        left_layout.addLayout(info_layout)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.setEnabled(False)
        self.slider.setFixedHeight(20)
        left_layout.addWidget(self.slider)

        # Nav buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        self.nav_buttons = []
        for label, delta in [("⏮ -100", -100), ("◀ -10", -10), ("◀ -1", -1),
                              ("▶ +1", 1), ("▶▶ +10", 10), ("▶▶▶ +100", 100)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size: 11px; border-radius: 4px; padding: 2px 6px;")
            btn.clicked.connect(lambda checked, d=delta: self._step(d))
            btn.setEnabled(False)
            self.nav_buttons.append(btn)
            btn_layout.addWidget(btn)
        left_layout.addLayout(btn_layout)

        splitter.addWidget(left)

        # ── Right: controls ────────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(370)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        # Legend
        legend = QGroupBox("Legend")
        leg_layout = QVBoxLayout(legend)
        leg_layout.setSpacing(4)
        leg_layout.addWidget(self._dot_label("CD1 skeleton (white mouse)", "#00008B"))
        leg_layout.addWidget(self._dot_label("BL6 skeleton (black mouse)", "#C80000"))
        leg_layout.addWidget(self._dot_label("CD1 'center' sample patch",  "#FFFFFF"))
        leg_layout.addWidget(self._dot_label("BL6 'center' sample patch",  "#FF8C00"))
        right_layout.addWidget(legend)

        # Suspected swap zones (color-based)
        self.zones_group = QGroupBox("Suspected Swap Zones (0)")
        zones_layout = QVBoxLayout(self.zones_group)
        self.zones_list = QListWidget()
        self.zones_list.setMaximumHeight(160)
        self.zones_list.setStyleSheet("font-size: 11px;")
        self.zones_list.itemClicked.connect(self._on_zone_click)
        zones_layout.addWidget(self.zones_list)
        btn_analyze = make_btn("🎨  (Re-)Run Color Analysis")
        btn_analyze.setFixedHeight(28)
        btn_analyze.clicked.connect(self._run_color_analysis)
        zones_layout.addWidget(btn_analyze)
        btn_autofix = make_btn("⚡  Auto-Fix All Detected Zones", color="#8e44ad", bold=True)
        btn_autofix.setFixedHeight(30)
        btn_autofix.clicked.connect(self._autofix_all_zones)
        zones_layout.addWidget(btn_autofix)
        right_layout.addWidget(self.zones_group)

        # Define swap range
        range_group = QGroupBox("Define Swap Range")
        range_layout = QVBoxLayout(range_group)
        range_layout.setSpacing(6)

        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Start:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        self.start_spin.setMaximum(999999)
        self.start_spin.setFixedHeight(26)
        spinbox_layout.addWidget(self.start_spin)
        spinbox_layout.addWidget(QLabel("End:"))
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(0)
        self.end_spin.setMaximum(999999)
        self.end_spin.setFixedHeight(26)
        spinbox_layout.addWidget(self.end_spin)
        range_layout.addLayout(spinbox_layout)

        btn_start = make_btn("📍  Set Start = Current Frame")
        btn_start.setFixedHeight(28)
        btn_start.clicked.connect(self._set_start)
        range_layout.addWidget(btn_start)

        btn_end = make_btn("📍  Set End = Current Frame")
        btn_end.setFixedHeight(28)
        btn_end.clicked.connect(self._set_end)
        range_layout.addWidget(btn_end)

        btn_apply = make_btn("🔄  Apply to change trajectories", color="#c0392b", bold=True)
        btn_apply.setFixedHeight(34)
        btn_apply.clicked.connect(self._apply_range_swap)
        range_layout.addWidget(btn_apply)
        right_layout.addWidget(range_group)

        # Applied swaps
        applied_group = QGroupBox("Applied Swaps")
        applied_layout = QVBoxLayout(applied_group)
        applied_layout.setSpacing(4)
        self.applied_list = QListWidget()
        self.applied_list.setMaximumHeight(120)
        self.applied_list.setStyleSheet("font-size: 11px;")
        applied_layout.addWidget(self.applied_list)
        btn_undo = make_btn("↩  Delete selected swap")
        btn_undo.setFixedHeight(28)
        btn_undo.clicked.connect(self._undo_swap)
        applied_layout.addWidget(btn_undo)
        right_layout.addWidget(applied_group)

        # Output + Save
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        out_layout.setSpacing(4)
        btn_set_out = make_btn("📁  Set Output path...")
        btn_set_out.setFixedHeight(28)
        btn_set_out.clicked.connect(self._set_output)
        out_layout.addWidget(btn_set_out)
        self.lbl_out = QLabel("Output: (auto)")
        self.lbl_out.setStyleSheet("color: #585b70; font-size: 10px;")
        self.lbl_out.setWordWrap(True)
        out_layout.addWidget(self.lbl_out)
        btn_save = make_btn("💾  Save Corrected csv file", color="#2d7a3a", bold=True)
        btn_save.setFixedHeight(36)
        btn_save.clicked.connect(self._save)
        out_layout.addWidget(btn_save)
        right_layout.addWidget(out_group)

        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([1100, 370])

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Load Tracking CSV and Video to start — color analysis runs automatically once both are loaded")

    def _dot_label(self, text, color):
        lbl = QLabel(f"⬤  {text}")
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        return lbl

    def _show_placeholder(self):
        img = np.zeros((480, 800, 3), dtype=np.uint8)
        text = "Open csv file and Video to Start"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (800 - tw) // 2
        y = (480 + th) // 2
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        self._display_image(img)

    # ── File loaders ───────────────────────────────────────────────────────────
    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open csv file", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            load_dlc(path)
            name = path.split('\\')[-1].split('/')[-1]
            self.lbl_csv.setText(f"CSV: {name}")
            self.lbl_csv.setStyleSheet("color: #a6e3a1; font-size: 10px;")
            self.lbl_out.setText(f"Output: {state['out_path']}")
            self._refresh_zones_list()
            self._enable_controls()
            self.status.showMessage(f"CSV loaded: {name}  |  {state['n_frames']} frames")
            self._maybe_run_color_analysis()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open original Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        if not path:
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self._cap_pos = -1
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            self.cap = None
            return
        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        name  = path.split('\\')[-1].split('/')[-1]
        self.lbl_video.setText(f"Video: {name}")
        self.lbl_video.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        self._enable_controls()
        self.status.showMessage(f"Video: {name}  |  {total} frames  |  {fps:.1f} fps")
        self._goto_frame(0)
        self._maybe_run_color_analysis()

    def _set_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Set Output CSV", state['out_path'] or "",
            "CSV files (*.csv);;All files (*)")
        if path:
            state['out_path'] = path
            self.lbl_out.setText(f"Output: {path}")

    # ── Color analysis / zones ───────────────────────────────────────────────────
    def _maybe_run_color_analysis(self):
        """Run the color-based swap analysis automatically once both the CSV
        and the video are loaded."""
        if state['df'] is not None and self.cap is not None:
            self._run_color_analysis()

    def _run_color_analysis(self):
        if state['df'] is None or self.cap is None:
            QMessageBox.warning(self, "Error", "Load both the Tracking CSV and the Video first")
            return
        self.status.showMessage("Analyzing frames for CD1/BL6 color mismatches…")
        QApplication.processEvents()

        def progress_cb(f, n):
            self.status.showMessage(f"Analyzing frames for color mismatches… {f}/{n}")
            QApplication.processEvents()

        try:
            analyze_color_swaps(self.cap, progress_cb=progress_cb)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Color analysis failed:\n{e}")
            return
        # analyze_color_swaps walked the whole video, so our cached read
        # position is no longer valid — force the next _goto_frame to re-seek.
        self._cap_pos = -1
        self._refresh_zones_list()
        n_flagged = int(state['color_flagged'].sum())
        self.status.showMessage(
            f"Color analysis done — {len(state['zones'])} suspected-swap zone(s), "
            f"{n_flagged} flagged frame(s) total"
        )
        self._goto_frame(self.current_frame)

    def _autofix_all_zones(self):
        zones = state['zones']
        if not zones:
            QMessageBox.information(self, "Info", "No suspected-swap zones to fix")
            return
        reply = QMessageBox.question(
            self, "Confirm Auto-Fix",
            f"This will swap CD1 ↔ BL6 for all {len(zones)} suspected zone(s) "
            f"(all bodyparts, not just 'center'). Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for zs, ze in zones:
            end_inclusive = ze - 1
            apply_swap_range(zs, end_inclusive)
            self.swap_ranges.append((zs, end_inclusive))
            item = QListWidgetItem(f"[{zs} – {end_inclusive}]   {end_inclusive - zs + 1} frames  (auto)")
            item.setData(Qt.UserRole, (zs, end_inclusive))
            self.applied_list.addItem(item)
        self.status.showMessage(f"✓ Auto-fixed {len(zones)} zone(s)")
        self._goto_frame(self.current_frame)

    def _refresh_zones_list(self):
        self.zones_list.clear()
        zones = state['zones']
        self.zones_group.setTitle(f"Suspected Swap Zones ({len(zones)})")
        for zs, ze in zones:
            item = QListWidgetItem(f"[{zs} – {ze - 1}]   {ze - zs} frames")
            item.setData(Qt.UserRole, (zs, ze))
            self.zones_list.addItem(item)
        n = state['n_frames']
        if n > 0:
            self.slider.setMaximum(n - 1)

    def _enable_controls(self):
        ready = state['df'] is not None and self.cap is not None
        self.slider.setEnabled(ready)
        for btn in self.nav_buttons:
            btn.setEnabled(ready)

    # ── Navigation ─────────────────────────────────────────────────────────────
    def _on_slider(self, value):
        # Coalesce rapid scrollbar movement. Render the first change immediately
        # (so it feels responsive), then throttle: further changes during the
        # throttle window just update the pending target, and only the newest is
        # rendered when the timer fires. Intermediate frames are dropped.
        self._pending_slider_frame = value
        if self._slider_render_timer.isActive():
            return
        self._pending_slider_frame = None
        self._goto_frame(value)
        self._slider_render_timer.start()

    def _render_pending_slider_frame(self):
        if self._pending_slider_frame is None:
            return  # no newer target since the last render
        value = self._pending_slider_frame
        self._pending_slider_frame = None
        self._goto_frame(value)
        self._slider_render_timer.start()  # keep the cadence going while dragging

    def _on_slider_released(self):
        # Make sure the exact frame the handle landed on is rendered, even if the
        # last movement got coalesced away.
        self._slider_render_timer.stop()
        pending = self._pending_slider_frame
        self._pending_slider_frame = None
        target = pending if pending is not None else self.slider.value()
        self._goto_frame(target)

    def _step(self, delta):
        n = state['n_frames']
        if n == 0:
            return
        self.slider.setValue(max(0, min(n - 1, self.current_frame + delta)))

    def _on_zone_click(self, item):
        zs, ze = item.data(Qt.UserRole)
        self.start_spin.setValue(zs)
        self.end_spin.setValue(ze - 1)
        self.slider.setValue(zs)

    def _set_start(self):
        self.start_spin.setValue(self.current_frame)
        self.status.showMessage(f"Start set → frame {self.current_frame}")

    def _set_end(self):
        self.end_spin.setValue(self.current_frame)
        self.status.showMessage(f"End set → frame {self.current_frame}")

    # ── Swap ───────────────────────────────────────────────────────────────────
    def _apply_range_swap(self):
        if state['df'] is None:
            QMessageBox.warning(self, "Error", "Load a Tracking CSV first")
            return
        start = self.start_spin.value()
        end   = self.end_spin.value()
        if end < start:
            QMessageBox.warning(self, "Error", "End frame must be ≥ Start frame")
            return
        apply_swap_range(start, end)
        self.swap_ranges.append((start, end))
        item = QListWidgetItem(f"[{start} – {end}]   {end - start + 1} frames")
        item.setData(Qt.UserRole, (start, end))
        self.applied_list.addItem(item)
        self.status.showMessage(f"✓ Swap applied: frames {start} – {end}")
        self._goto_frame(self.current_frame)

    def _undo_swap(self):
        selected = self.applied_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "Info", "Select a swap from the list to undo")
            return
        item = selected[0]
        start, end = item.data(Qt.UserRole)
        undo_swap_range(start, end)
        self.applied_list.takeItem(self.applied_list.row(item))
        self.status.showMessage(f"↩ Undo: frames {start} – {end}")
        self._goto_frame(self.current_frame)

    # ── Frame rendering ────────────────────────────────────────────────────────
    def _read_frame_at(self, frame_idx):
        """Return the decoded image for exactly `frame_idx`.

        Avoids the 'skeleton lags behind video' problem: with many codecs
        cap.set(POS_FRAMES, n) only lands on the nearest keyframe, so the
        displayed image can be a few frames off from the CSV row we draw.
        Strategy:
          * If we're moving one frame forward from the last frame we read,
            just read the next frame — OpenCV decodes sequentially and is
            frame-accurate this way (no seek at all).
          * Otherwise seek, then check the real position OpenCV reports and
            read forward until we reach the requested frame, so the returned
            image truly matches frame_idx.
        """
        cap = self.cap

        # Fast path: exact next frame → no seek, guaranteed in-sync.
        if self._cap_pos == frame_idx - 1:
            ret, img = cap.read()
            if ret:
                self._cap_pos = frame_idx
                return img
            # fall through to seek-based path if the plain read failed

        # Seek, then correct for keyframe-snapping by reading forward.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # If the decoder landed before our target (keyframe snap), read forward.
        # Cap the catch-up work so a pathological video can't freeze the UI.
        img = None
        if actual <= frame_idx:
            steps = 0
            max_steps = 300
            while actual <= frame_idx and steps < max_steps:
                ret, frame = cap.read()
                if not ret:
                    break
                img = frame
                actual += 1
                steps += 1
            self._cap_pos = actual - 1
        else:
            # Decoder reported a position past the target; do a plain read.
            ret, img = cap.read()
            self._cap_pos = frame_idx if ret else -1

        return img

    def _goto_frame(self, frame_idx):
        if self.cap is None or state['df'] is None:
            return
        self.current_frame = frame_idx
        img = self._read_frame_at(frame_idx)
        if img is None:
            return

        n            = state['n_frames']
        has_analysis = len(state['color_flagged']) > 0

        if frame_idx < n:
            kps_cd1 = get_keypoints(frame_idx, state['cd1_map'])
            kps_bl6 = get_keypoints(frame_idx, state['bl6_map'])
            draw_skeleton(img, kps_cd1, COLOR_CD1)
            draw_skeleton(img, kps_bl6, COLOR_BL6)

            x1, y1, _ = get_center_xy_likelihood(frame_idx, state['cd1_map'])
            x2, y2, _ = get_center_xy_likelihood(frame_idx, state['bl6_map'])
            draw_sample_patch(img, x1, y1, args.patch_radius, COLOR_SAMPLE_CD1)
            draw_sample_patch(img, x2, y2, args.patch_radius, COLOR_SAMPLE_BL6)

            if has_analysis and frame_idx < len(state['color_flagged']) and state['color_flagged'][frame_idx]:
                cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1),
                              COLOR_FLAGGED, 6)
                cv2.putText(img, "SUSPECTED SWAP", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_FLAGGED, 3)
                self.zone_label.setText("⚠  SUSPECTED SWAP")
            else:
                self.zone_label.setText("")

        cv2.putText(img, f"Frame {frame_idx}", (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        self._display_image(img)

        bright_str = ""
        if has_analysis and frame_idx < len(state['brightness_cd1']):
            b1 = state['brightness_cd1'][frame_idx]
            b2 = state['brightness_bl6'][frame_idx]
            if not (np.isnan(b1) or np.isnan(b2)):
                bright_str = f"   |   CD1: {b1:.0f}  BL6: {b2:.0f}"
        self.frame_label.setText(f"Frame  {frame_idx} / {n - 1}{bright_str}")

        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

    def _display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    def _save(self):
        if state['df'] is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        out = state['out_path']
        if not out:
            QMessageBox.warning(self, "Error", "Set an output path first")
            return
        state['df'].to_csv(out)
        QMessageBox.information(self, "Saved",
                                f"File saved:\n{out}\n\n"
                                f"Swap ranges applied: {len(self.swap_ranges)}")
        self.status.showMessage(f"✓ Saved: {out}")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()


# ── Run ────────────────────────────────────────────────────────────────────────
app = QApplication(sys.argv)
app.setStyleSheet(DARK_STYLE)
app.setWindowIcon(make_mouse_icon())

# Splash screen with the full brand image (mouse + "TrackFix" + subtext) on black background
splash = TrackFixSplashScreen()
splash.show()
app.processEvents()

win = MainWindow()

def _finish_splash():
    splash.close()
    win.show()

# Keep the splash on screen for SPLASH_DURATION_MS, then swap to the main window.
# Clicking the splash also dismisses it early.
splash.mousePressEvent = lambda event: _finish_splash()
QTimer.singleShot(SPLASH_DURATION_MS, _finish_splash)

sys.exit(app.exec_())