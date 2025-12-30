import threading
import time
import numpy as np
import pygame
import cv2
from config import *

# for use in windows testing
import unittest
from unittest.mock import patch, MagicMock


# ============================
# Bayer 4x4 (ordered dithering)
# ============================

BAYER_4x4 = (1 / 17) * np.array([
    [0,  8,  2, 10],
    [12, 4, 14, 6],
    [3, 11, 1,  9],
    [15, 7, 13, 5]
], dtype=np.float32)

# ============================
# Shared state
# ============================

latest_luma = None
frame_lock = threading.Lock()
running = True
import threading
import time
import numpy as np
import pygame
import cv2
from config import *


# Attempt to import Picamera2; provide a Dummy fallback for Windows/testing
try:
    from picamera2 import Picamera2
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


class DummyRequest:
    def __init__(self, arr):
        self._arr = arr

    def make_array(self, key):
        return self._arr

    def release(self):
        return


class DummyPicamera2:
    def __init__(self):
        self._running = False

    def create_video_configuration(self, *args, **kwargs):
        return {}

    def configure(self, cfg):
        return

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def set_controls(self, controls):
        return

    def capture_request(self):
        # produce a synthetic luma frame so preview works on non-camera systems
        t = int((time.time() * 1000) % 256)
        arr = np.full((CAM_H, CAM_W), t, dtype=np.uint8)
        return DummyRequest(arr)


class Application:
    def __init__(self):
        # Shared state
        self.latest_luma = None
        self.frame_lock = threading.Lock()
        self.running = True

        # UI-controlled parameters
        self.exposure_us = 3000
        self.mask_cutoff = 120
        self.dither_enable = 0

        # Camera
        self.picam2 = Picamera2() if Picamera2 is not None else DummyPicamera2()
        try:
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
        except Exception:
            # Some dummy/config objects may not accept configure; ignore non-fatal errors
            pass

        self.picam2.start()

    # Camera capture loop
    def _camera_loop(self):
        while self.running:
            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
                y_plane = frame[:CAM_H, :CAM_W]
                with self.frame_lock:
                    self.latest_luma = y_plane
            finally:
                try:
                    request.release()
                except Exception:
                    pass

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

        font = pygame.font.SysFont(None, 22)
        text = font.render(f"{label}: {value}", True, (255, 255, 255))
        screen.blit(text, (x, y - 18))

        return track

    # Preview + UI loop
    def preview_loop(self):
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Camera + Inverted Mask Preview")
        clock = pygame.time.Clock()

        active_slider = None
        exposure_rect = cutoff_rect = dither_rect = pygame.Rect(0, 0, 0, 0)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if exposure_rect.collidepoint(mx, my):
                        active_slider = "exposure"
                    elif cutoff_rect.collidepoint(mx, my):
                        active_slider = "cutoff"
                    elif dither_rect.collidepoint(mx, my):
                        active_slider = "dither"

                elif event.type == pygame.MOUSEBUTTONUP:
                    active_slider = None

                elif event.type == pygame.MOUSEMOTION and active_slider:
                    mx, _ = event.pos
                    t = np.clip((mx - SLIDER_X) / SLIDER_W, 0, 1)

                    if active_slider == "exposure":
                        self.exposure_us = int(500 + t * 5500)
                        try:
                            self.picam2.set_controls({
                                "AeEnable": False,
                                "ExposureTime": self.exposure_us
                            })
                        except Exception:
                            pass

                    elif active_slider == "cutoff":
                        self.mask_cutoff = int(t * 255)

                    elif active_slider == "dither":
                        self.dither_enable = 1 if t > 0.5 else 0

            with self.frame_lock:
                if self.latest_luma is None:
                    time.sleep(0.001)
                    continue
                frame = self.latest_luma.copy()

            cam_rgb = np.stack([frame] * 3, axis=-1)
            cam_surf = pygame.surfarray.make_surface(np.rot90(cam_rgb))

            lcd_luma = cv2.resize(frame, (LCD_W, LCD_H), interpolation=cv2.INTER_AREA)
            mask = (lcd_luma <= self.mask_cutoff).astype(np.uint8) * 255
            mask_rgb = np.stack([mask] * 3, axis=-1)
            mask_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(np.rot90(mask_rgb)),
                (CAM_W, CAM_H)
            )

            screen.fill((0, 0, 0))
            screen.blit(cam_surf, (0, 0))
            screen.blit(mask_surf, (0, CAM_H))

            exposure_rect = self.draw_slider(
                screen, SLIDER_X, SLIDER_Y_EXPOSURE,
                SLIDER_W, SLIDER_H,
                self.exposure_us, 500, 6000, "Exposure (us)"
            )

            cutoff_rect = self.draw_slider(
                screen, SLIDER_X, SLIDER_Y_CUTOFF,
                SLIDER_W, SLIDER_H,
                self.mask_cutoff, 0, 255, "Mask Cutoff"
            )

            dither_rect = self.draw_slider(
                screen, SLIDER_X, SLIDER_Y_DITHER,
                SLIDER_W, SLIDER_H,
                self.dither_enable, 0, 1, "Dithering"
            )

            pygame.display.flip()
            clock.tick(PREVIEW_FPS)

        pygame.quit()

    def start(self):
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._lcd_thread = threading.Thread(target=self._lcd_loop, daemon=True)
        self._camera_thread.start()
        self._lcd_thread.start()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        try:
            self.picam2.stop()
        except Exception:
            pass


def main():
    app = Application()
    try:
        app.start()
        app.preview_loop()
    finally:
        app.stop()


if __name__ == '__main__':
    main()
