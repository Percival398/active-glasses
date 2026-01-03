import contextlib
import sys
import time
import numpy as np
import pygame
import cv2
from config import *
from dummyCam import DummyPicamera2

def lcd_loop(app):
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Mask Preview (fullscreen)")

    while app.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.running = False
                pygame.quit()
                sys.exit()

            # Check for a key press
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    app.running = False
                    pygame.quit()
                    sys.exit()
                    return

                
        with app.frame_lock:
            if app.latest_luma is None:
                time.sleep(0.001)
                continue
            frame = app.latest_luma.copy()
        try:
            app._cam_rgb_buf[..., 0] = frame
            app._cam_rgb_buf[..., 1] = frame
            app._cam_rgb_buf[..., 2] = frame
            cam_surf = pygame.surfarray.make_surface(np.rot90(app._cam_rgb_buf))
        except Exception:
            cam_rgb = np.stack([frame] * 3, axis=-1)
            cam_surf = pygame.surfarray.make_surface(np.rot90(cam_rgb))
        lcd_luma = cv2.resize(frame, (LCD_W, LCD_H), interpolation=cv2.INTER_AREA)
    
        try:
            _, mask = cv2.threshold(lcd_luma, int(app.mask_cutoff), 255, cv2.THRESH_BINARY_INV)
        except Exception:
            mask = (lcd_luma <= app.mask_cutoff).astype(np.uint8) * 255

        if getattr(app, 'denoise_h', 0) and app.denoise_h > 0.0:
            try:
                den = cv2.fastNlMeansDenoising(mask, None, h=float(app.denoise_h), templateWindowSize=7, searchWindowSize=21)
                mask = den
            except Exception:
                mask = cv2.medianBlur(mask, 3)

        if getattr(app, 'expand_px', 0) and app.expand_px > 0:
            k = max(1, int(app.expand_px))
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            mask = cv2.erode(mask, kern, iterations=1)

        if getattr(app, 'post_blur', 0) and app.post_blur > 0.0:
            radius = int(max(0, round(app.post_blur)))
            ksize = radius * 2 + 1
            if ksize > 1:
                with contextlib.suppress(Exception):
                    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        try:
            app._mask_rgb[..., 0] = mask
            app._mask_rgb[..., 1] = mask
            app._mask_rgb[..., 2] = mask
            mask_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(np.rot90(app._mask_rgb)),
                (CAM_W, CAM_H)
            )
        except Exception:
            mask_rgb = np.stack([mask] * 3, axis=-1)
            mask_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(np.rot90(mask_rgb)),
                (CAM_W, CAM_H)
            )
            
        # display the thing
        screen_w, screen_h = screen.get_size()
        full_scaled = pygame.transform.smoothscale(mask_surf, (screen_w, screen_h))
        screen.fill((0, 0, 0))
        screen.blit(full_scaled, (0, 0))
        pygame.display.flip()
        
    pygame.quit()
