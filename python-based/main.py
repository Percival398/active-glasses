
import contextlib
import argparse
import threading
import time
import os
import numpy as np
import pygame
import cv2
from config import *
from dummyCam import *  # Ensure dummyCamera2 is imported for fallback
from preview.preview import preview_loop as external_preview_loop

latest_luma = None
frame_lock = threading.Lock()
running = True

# Attempt to import Picamera2; provide a Dummy fallback for Windows/testing
try:
    from picamera2 import Picamera2 # type: ignore
except Exception as e:
    Picamera2 = None
    print("picamera2 not found — using DummyPicamera2 fallback (Windows/testing).")


# Bayer 4x4 (ordered dithering)
BAYER_4x4 = (1 / 17) * np.array([
    [0,  8,  2, 10],
    [12, 4, 14, 6],
    [3, 11, 1,  9],
    [15, 7, 13, 5]
], dtype=np.float32)

class Application:
    def __init__(self, mask_only=True):
        # Shared state
        self.latest_luma = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Mask-only mode (display only mask, no labels/sliders)
        self.mask_only = mask_only

        # UI-controlled parameters driven from `SLIDERS` in config.py
        self.slider_defs = SLIDERS
        # initialize attributes from slider defaults
        for s in self.slider_defs:
            setattr(self, s['key'], s.get('default'))
        # quick lookup and runtime UI rects
        self.slider_map = {s['key']: s for s in self.slider_defs}
        self.slider_rects = {}

        # Camera
        self.picam2 = DummyPicamera2()
        with contextlib.suppress(Exception):
            self.picam2.configure(
                self.picam2.create_video_configuration(
                    main={
                        "size": (CAM_W, CAM_H),
                        "format": "YUV420"
                    },
                    buffer_count=6,
                    controls={
                        "FrameRate": CAM_FPS,
                        "AeEnable": False,
                        "ExposureTime": self.exposure_us
                    }
                )
            )
        self.picam2.start()
        self._test_img = None
        # preview timing history for estimating frame time / fps
        self._preview_prev_time = None
        self._preview_time_hist = []
        # Precompute Bayer tiled threshold (0..255) for LCD size and allocate RGB buffers
        try:
            self._bayer_tiled = (np.tile(BAYER_4x4, (LCD_H // 4 + 1, LCD_W // 4 + 1))[:LCD_H, :LCD_W] * 255).astype(np.uint8)
        except Exception:
            self._bayer_tiled = None
        self._mask_rgb = np.empty((LCD_H, LCD_W, 3), dtype=np.uint8)
        self._cam_rgb_buf = np.empty((CAM_H, CAM_W, 3), dtype=np.uint8)
        # preferred monospace font name (create font after pygame.init)
        self._mono_font_name = "Courier New"
        self._mono_font_size = 20

    # Camera capture loop
    def _camera_loop(self):
        while self.running:
            try:
                request = self.picam2.capture_request()
            except Exception:
                # camera not working; use test image fallback
                y_plane = self._load_test_image()
                if y_plane is not None:
                    with self.frame_lock:
                        self.latest_luma = y_plane
                time.sleep(0.01)
                continue

            try:
                frame = request.make_array("main")
            except Exception:
                with contextlib.suppress(Exception):
                    request.release()
                y_plane = self._load_test_image()
                if y_plane is not None:
                    with self.frame_lock:
                        self.latest_luma = y_plane
                time.sleep(0.01)
                continue

            try:
                # camera frame may be color; ensure we keep luma
                if frame is None:
                    y_plane = self._load_test_image()
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    y_plane = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    y_plane = frame[:CAM_H, :CAM_W]

                with self.frame_lock:
                    self.latest_luma = y_plane
            finally:
                with contextlib.suppress(Exception):
                    request.release()

    # LCD processing loop (keeps timing to camera FPS)
    def _lcd_loop(self):
        frame_period = 1.0 / CAM_FPS
        while self.running:
            start = time.perf_counter()

            with self.frame_lock:
                if self.latest_luma is None:
                    time.sleep(0.001)
                    continue
                luma = self.latest_luma

            lcd_luma = cv2.resize(luma, (LCD_W, LCD_H), interpolation=cv2.INTER_AREA)

            if self.dither_enable:
                # use precomputed Bayer tiled thresholds when available
                if getattr(self, '_bayer_tiled', None) is not None:
                    try:
                        lcd_luma = (lcd_luma > self._bayer_tiled) * 255
                    except Exception:
                        tiled = np.tile(BAYER_4x4, (LCD_H // 4 + 1, LCD_W // 4 + 1))[:LCD_H, :LCD_W]
                        lcd_norm = lcd_luma / 255.0
                        lcd_luma = (lcd_norm > tiled) * 255
                else:
                    tiled = np.tile(BAYER_4x4, (LCD_H // 4 + 1, LCD_W // 4 + 1))[:LCD_H, :LCD_W]
                    lcd_norm = lcd_luma / 255.0
                    lcd_luma = (lcd_norm > tiled) * 255

            lcd_mask = (lcd_luma <= self.mask_cutoff).astype(np.uint8) * 255
            # Here you'd send `lcd_mask` to the hardware LCD if available

            elapsed = time.perf_counter() - start
            sleep = frame_period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    # UI helper for drawing a slider
    def draw_slider(self, screen, x, y, w, h, value, min_v, max_v, label):
        track = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, (90, 90, 90), track, border_radius=4)

        t = (value - min_v) / (max_v - min_v)
        knob_x = int(x + t * w)
        pygame.draw.circle(screen, (230, 230, 230), (knob_x, y + h // 2), h)

        font = pygame.font.SysFont(self._mono_font_name, 22)
        # nicely format numeric value (show one decimal when fractional)
        try:
            fv = float(value)
            vs = f"{int(fv)}" if fv.is_integer() else f"{fv:.1f}"
        except Exception:
            vs = str(value)
        text = font.render(f"{label}: {vs}", True, (255, 255, 255))
        screen.blit(text, (x, y - 18))

        return track

    # Preview + UI loop
    # Preview loop delegated to external preview module
    def preview_loop(self):
        return external_preview_loop(self)

    def start(self):
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._lcd_thread = threading.Thread(target=self._lcd_loop, daemon=True)
        self._camera_thread.start()
        self._lcd_thread.start()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        with contextlib.suppress(Exception):
            self.picam2.stop()

    def _load_test_image(self):
        # cached load of test image from disk (grayscale)
        if self._test_img is not None:
            return self._test_img

        with contextlib.suppress(Exception):
            base = os.path.dirname(__file__)
            path = os.path.join(base, 'preview/test_img02.jpg')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                    self._test_img = img
                    print(f"Loaded test image from {path} for fallback.")
                    return self._test_img
        # final fallback: synthetic gradient-like frame
        arr = np.full((CAM_H, CAM_W), 128, dtype=np.uint8)
        self._test_img = arr
        return self._test_img


def main():
    parser = argparse.ArgumentParser(description="Active glasses preview")
    parser.add_argument('--mask-only', action='store_true', help='Show only mask window (no labels/sliders) in fullscreen')
    args = parser.parse_args()

    app = Application(mask_only=args.mask_only)
    try:
        app.start()
        app.preview_loop()
    finally:
        app.stop()


if __name__ == '__main__':
    main()
