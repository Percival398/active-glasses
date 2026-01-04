import contextlib
import sys
import time
import numpy as np
import pygame
import cv2
from config import *
from dummyCam import *

def _handle_events(app):
    """Handle pygame window and keyboard events."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _leave(app)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            _leave(app)


def _leave(app):
    app.running = False
    pygame.quit()
    sys.exit()


def _compute_mask(app, frame):
    """Compute the mask image from the luma frame."""
    lcd_luma = cv2.resize(frame, (LCD_W, LCD_H), interpolation=cv2.INTER_AREA)

    try:
        _, mask = cv2.threshold(
            lcd_luma,
            int(app.mask_cutoff),
            255,
            cv2.THRESH_BINARY_INV,
        )
    except Exception:
        mask = (lcd_luma <= app.mask_cutoff).astype(np.uint8) * 255

    if getattr(app, "denoise_h", 0) and app.denoise_h > 0.0:
        try:
            mask = cv2.fastNlMeansDenoising(
                mask,
                None,
                h=float(app.denoise_h),
                templateWindowSize=7,
                searchWindowSize=21,
            )
        except Exception:
            mask = cv2.medianBlur(mask, 3)

    if getattr(app, "expand_px", 0) and app.expand_px > 0:
        k = max(1, int(app.expand_px))
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * k + 1, 2 * k + 1),
        )
        mask = cv2.erode(mask, kern, iterations=1)

    if getattr(app, "post_blur", 0) and app.post_blur > 0.0:
        radius = int(max(0, round(app.post_blur)))
        ksize = radius * 2 + 1
        if ksize > 1:
            with contextlib.suppress(Exception):
                mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask


def _mask_surface_from_mask(app, mask):
    """Convert a mask array into a pygame surface scaled to camera size."""
    try:
        app._mask_rgb[..., 0] = mask
        app._mask_rgb[..., 1] = mask
        app._mask_rgb[..., 2] = mask
        src = app._mask_rgb
    except Exception:
        src = np.stack([mask] * 3, axis=-1)

    return pygame.transform.scale(
        pygame.surfarray.make_surface(np.rot90(src)),
        (CAM_W, CAM_H),
    )


def _display_fullscreen(screen, mask_surf):
    """Display the mask surface fullscreen."""
    screen_w, screen_h = screen.get_size()
    full_scaled = pygame.transform.smoothscale(mask_surf, (screen_w, screen_h))
    screen.fill((0, 0, 0))
    screen.blit(full_scaled, (0, 0))
    pygame.display.flip()


def lcd_loop(app):
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (info.current_w, info.current_h),
        pygame.FULLSCREEN,
    )
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Mask Preview (fullscreen)")

    while app.running:
        _handle_events(app)

        with app.frame_lock:
            if app.latest_luma is None:
                time.sleep(0.001)
                continue
            frame = app.latest_luma.copy()

        mask = _compute_mask(app, frame)
        mask_surf = _mask_surface_from_mask(app, mask)
        mask_surf_flipped = pygame.transform.flip(mask_surf, False, True)
        _display_fullscreen(screen, mask_surf_flipped)

    pygame.quit()
