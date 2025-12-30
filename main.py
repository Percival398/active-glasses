
import threading
import time
import os
import numpy as np
import pygame
import cv2
from config import *
from dummyCam import *  # Ensure dummyCamera2 is imported for fallback

latest_luma = None
frame_lock = threading.Lock()
running = True

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
        # New mask-processing parameters
        self.denoise_h = 6.0            # NL-means filter strength (0 = off)
        self.expand_px = 0              # morphological expansion in pixels
        self.post_blur = 0.0            # gaussian blur kernel radius (0 = off)

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
                try:
                    request.release()
                except Exception:
                    pass
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
            if fv.is_integer():
                vs = f"{int(fv)}"
            else:
                vs = f"{fv:.1f}"
        except Exception:
            vs = str(value)
        text = font.render(f"{label}: {vs}", True, (255, 255, 255))
        screen.blit(text, (x, y - 18))

        return track

    # Preview + UI loop
    def preview_loop(self):
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        pygame.display.set_caption("Camera + Inverted Mask Preview")
        clock = pygame.time.Clock()
        # create monospace fonts once to avoid per-frame allocations
        try:
            font_small = pygame.font.SysFont(self._mono_font_name, self._mono_font_size)
        except Exception:
            font_small = pygame.font.SysFont(None, self._mono_font_size)

        active_slider = None
        exposure_rect = cutoff_rect = dither_rect = denoise_rect = expand_rect = blur_rect = pygame.Rect(0, 0, 0, 0)

        while self.running:
            # record preview-loop timing for a short moving average used to estimate frame time/fps
            now = time.perf_counter()
            if self._preview_prev_time is None:
                dt = 1.0 / CAM_FPS
            else:
                dt = now - self._preview_prev_time
            self._preview_prev_time = now
            self._preview_time_hist.append(dt)
            if len(self._preview_time_hist) > 32:
                self._preview_time_hist.pop(0)

            # compute window-relative layout so image stack stays left and sliders sit to the right
            win_w, win_h = screen.get_size()
            slider_panel_w = SLIDER_W + 40
            max_image_w = max(100, win_w - slider_panel_w - 20)
            combined_aspect = CAM_W / (CAM_H * 2)
            target_w = min(max_image_w, int(win_h * combined_aspect))
            if target_w < 1:
                target_w = 1
            target_h = max(1, int(target_w / combined_aspect))
            blit_x = 10
            blit_y = max(0, (win_h - target_h) // 2)
            slider_x = blit_x + target_w + 20

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
                    elif denoise_rect.collidepoint(mx, my):
                        active_slider = "denoise"
                    elif expand_rect.collidepoint(mx, my):
                        active_slider = "expand"
                    elif blur_rect.collidepoint(mx, my):
                        active_slider = "blur"

                elif event.type == pygame.MOUSEBUTTONUP:
                    active_slider = None

                elif event.type == pygame.MOUSEMOTION and active_slider:
                    mx, _ = event.pos
                    t = np.clip((mx - slider_x) / SLIDER_W, 0, 1)

                    match active_slider:
                        case "exposure":
                            self.exposure_us = int(500 + t * 5500)
                            try:
                                self.picam2.set_controls({
                                    "AeEnable": False,
                                    "ExposureTime": self.exposure_us
                                })
                            except Exception:
                                pass
                        case "cutoff":
                            self.mask_cutoff = int(t * 255)
                        case "dither":
                            self.dither_enable = 1 if t > 0.5 else 0
                        case "denoise":
                            # map t -> 0..30
                            self.denoise_h = float(t * 30.0)
                        case "expand":
                            # map t -> 0..24 px
                            self.expand_px = int(t * 24)
                        case "blur":
                            # map t -> 0..25 px gaussian kernel radius
                            self.post_blur = float(t * 25.0)

            with self.frame_lock:
                if self.latest_luma is None:
                    time.sleep(0.001)
                    continue
                frame = self.latest_luma.copy()

            # reuse preallocated RGB buffer to reduce allocations
            try:
                self._cam_rgb_buf[..., 0] = frame
                self._cam_rgb_buf[..., 1] = frame
                self._cam_rgb_buf[..., 2] = frame
                cam_surf = pygame.surfarray.make_surface(np.rot90(self._cam_rgb_buf))
            except Exception:
                cam_rgb = np.stack([frame] * 3, axis=-1)
                cam_surf = pygame.surfarray.make_surface(np.rot90(cam_rgb))

            lcd_luma = cv2.resize(frame, (LCD_W, LCD_H), interpolation=cv2.INTER_AREA)
            # initial binary mask from threshold (use OpenCV threshold for speed)
            try:
                _, mask = cv2.threshold(lcd_luma, int(self.mask_cutoff), 255, cv2.THRESH_BINARY_INV)
            except Exception:
                mask = (lcd_luma <= self.mask_cutoff).astype(np.uint8) * 255

            # 1) optional denoise on mask (helps remove speckle). use NL-means on 8-bit mask
            if getattr(self, 'denoise_h', 0) and self.denoise_h > 0.0:
                try:
                    den = cv2.fastNlMeansDenoising(mask, None, h=float(self.denoise_h), templateWindowSize=7, searchWindowSize=21)
                    mask = den
                except Exception:
                    # fallback: small median blur
                    mask = cv2.medianBlur(mask, 3)

            # 2) shrink mask edges by morphological erosion (was dilation)
            if getattr(self, 'expand_px', 0) and self.expand_px > 0:
                k = max(1, int(self.expand_px))
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                mask = cv2.erode(mask, kern, iterations=1)

            # 3) optional post-expansion blur to smooth transition into white
            if getattr(self, 'post_blur', 0) and self.post_blur > 0.0:
                # kernel size must be odd; derive from radius
                radius = int(max(0, round(self.post_blur)))
                ksize = radius * 2 + 1
                if ksize > 1:
                    try:
                        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
                    except Exception:
                        pass
            # fill reusable RGB mask buffer and make surface (then scale to CAM size)
            try:
                self._mask_rgb[..., 0] = mask
                self._mask_rgb[..., 1] = mask
                self._mask_rgb[..., 2] = mask
                mask_surf = pygame.transform.scale(
                    pygame.surfarray.make_surface(np.rot90(self._mask_rgb)),
                    (CAM_W, CAM_H)
                )
            except Exception:
                mask_rgb = np.stack([mask] * 3, axis=-1)
                mask_surf = pygame.transform.scale(
                    pygame.surfarray.make_surface(np.rot90(mask_rgb)),
                    (CAM_W, CAM_H)
                )

            # Composite camera and mask into a single surface and scale to the computed image area
            combined = pygame.Surface((CAM_W, CAM_H * 2))
            combined.blit(cam_surf, (0, 0))
            combined.blit(mask_surf, (0, CAM_H))

            # Draw labels for the two preview frames (bottom-left of each frame)
            try:
                simulated = isinstance(self.picam2, DummyPicamera2)
            except Exception:
                simulated = False

            top_label = "Camera (simulated)" if simulated else "Camera"
            bottom_label = "LCD output"

            # Use monospace font for labels/timing to keep box widths stable
            font = font_small

            # Helper to draw a label with a semi-transparent black box
            # If fixed_w is provided, the background box will use that width
            def draw_label(surface, text, x, y, padding=6, alpha=160, fixed_w=None):
                txt = font.render(text, True, (255, 255, 255))
                w, h = txt.get_width(), txt.get_height()
                box_w = (fixed_w if fixed_w is not None else w) + padding * 2
                bg = pygame.Surface((box_w, h + padding * 2), pygame.SRCALPHA)
                bg.fill((0, 0, 0, alpha))
                surface.blit(bg, (x - padding, y - padding))
                # draw text left-aligned within the fixed box
                surface.blit(txt, (x, y))

            margin = 6

            # Top frame bottom-left (inside the top frame)
            txt_top = top_label
            top_x = margin
            top_y = CAM_H - font.get_height() - margin
            # compute fixed width for frame labels (use the longest possible label)
            max_label_text = "Camera (simulated)"
            label_fixed_w = font.size(max_label_text)[0]
            draw_label(combined, txt_top, top_x, top_y, fixed_w=label_fixed_w)

            # Bottom frame bottom-left (inside the bottom frame)
            txt_bot = bottom_label
            bot_x = margin
            bot_y = CAM_H * 2 - font.get_height() - margin
            draw_label(combined, txt_bot, bot_x, bot_y, fixed_w=label_fixed_w)

            # Bottom frame bottom-right: estimated frame time and fps (moving average)
            if len(self._preview_time_hist) > 0:
                avg = sum(self._preview_time_hist) / len(self._preview_time_hist)
            else:
                avg = 1.0 / CAM_FPS
            microsec = avg * 1e3
            fps = int(round(1.0 / avg)) if avg > 0 else 0
            ft = f"{microsec:5.1f}ms ({fps:03d} fps)"

            # Render timing text using a fixed-width background so it doesn't resize
            sample = f"{9999999.9:07.1f}us ({999:03d} fps)"
            fixed_w = font.size(sample)[0]
            txt = font.render(ft, True, (255, 255, 255))
            h = txt.get_height()
            padding = 6
            alpha = 160
            tx = CAM_W - margin - fixed_w
            ty = CAM_H * 2 - h - margin
            bg = pygame.Surface((fixed_w + padding * 2, h + padding * 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, alpha))
            combined.blit(bg, (tx - padding, ty - padding))
            # right-align the actual text inside the fixed box
            text_x = tx + (fixed_w - txt.get_width())
            combined.blit(txt, (text_x, ty))

            scaled = pygame.transform.smoothscale(combined, (target_w, target_h))
            screen.fill((0, 0, 0))
            screen.blit(scaled, (blit_x, blit_y))

            # place sliders vertically to the right of the image stack
            slider_y0 = blit_y + 20
            spacing = 44
            exposure_rect = self.draw_slider(
                screen, slider_x, slider_y0,
                SLIDER_W, SLIDER_H,
                self.exposure_us, 500, 6000, "Exposure (us)"
            )

            cutoff_rect = self.draw_slider(
                screen, slider_x, slider_y0 + spacing,
                SLIDER_W, SLIDER_H,
                self.mask_cutoff, 0, 255, "Mask Cutoff"
            )

            dither_rect = self.draw_slider(
                screen, slider_x, slider_y0 + spacing * 2,
                SLIDER_W, SLIDER_H,
                self.dither_enable, 0, 1, "Dithering"
            )

            denoise_rect = self.draw_slider(
                screen, slider_x, slider_y0 + spacing * 3,
                SLIDER_W, SLIDER_H,
                self.denoise_h, 0.0, 30.0, "Denoise H"
            )

            expand_rect = self.draw_slider(
                screen, slider_x, slider_y0 + spacing * 4,
                SLIDER_W, SLIDER_H,
                self.expand_px, 0, 24, "Edge Grow"
            )

            blur_rect = self.draw_slider(
                screen, slider_x, slider_y0 + spacing * 5,
                SLIDER_W, SLIDER_H,
                self.post_blur, 0.0, 25.0, "Post Blur"
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

    def _load_test_image(self):
        # cached load of test image from disk (grayscale)
        if self._test_img is not None:
            return self._test_img

        try:
            base = os.path.dirname(__file__)
            path = os.path.join(base, 'preview/test_img02.jpg')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                    self._test_img = img
                    print(f"Loaded test image from {path} for fallback.")
                    return self._test_img
        except Exception:
            pass

        # final fallback: synthetic gradient-like frame
        arr = np.full((CAM_H, CAM_W), 128, dtype=np.uint8)
        self._test_img = arr
        return self._test_img


def main():
    app = Application()
    try:
        app.start()
        app.preview_loop()
    finally:
        app.stop()


if __name__ == '__main__':
    main()
