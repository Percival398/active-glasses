import contextlib
import time
import numpy as np
import pygame
import cv2
from config import *
from dummyCam import DummyPicamera2


def draw_slider(app, screen, x, y, w, h, value, min_v, max_v, label):
    track = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (90, 90, 90), track, border_radius=4)

    t = (value - min_v) / (max_v - min_v) if (max_v - min_v) != 0 else 0
    knob_x = int(x + t * w)
    pygame.draw.circle(screen, (230, 230, 230), (knob_x, y + h // 2), h)

    try:
        font = pygame.font.SysFont(app._mono_font_name, 22)
    except Exception:
        font = pygame.font.SysFont(None, 22)

    try:
        fv = float(value)
        vs = f"{int(fv)}" if fv.is_integer() else f"{fv:.1f}"
    except Exception:
        vs = str(value)
    text = font.render(f"{label}: {vs}", True, (255, 255, 255))
    screen.blit(text, (x, y - 18))

    return track


def preview_loop(app):
    pygame.init()
    # If mask-only mode requested, open a fullscreen window; otherwise a resizable window
    if getattr(app, 'mask_only', False):
        info = pygame.display.Info()
        screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        pygame.display.set_caption("Mask Preview (fullscreen)")
    else:
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        pygame.display.set_caption("Camera + Inverted Mask Preview")
    clock = pygame.time.Clock()
    try:
        font_small = pygame.font.SysFont(app._mono_font_name, app._mono_font_size)
    except Exception:
        font_small = pygame.font.SysFont(None, app._mono_font_size)

    active_slider = None

    while app.running:
        now = time.perf_counter()
        if app._preview_prev_time is None:
            dt = 1.0 / CAM_FPS
        else:
            dt = now - app._preview_prev_time
        app._preview_prev_time = now
        app._preview_time_hist.append(dt)
        if len(app._preview_time_hist) > 32:
            app._preview_time_hist.pop(0)

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
                app.running = False
                return

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for key, rect in app.slider_rects.items():
                    if rect.collidepoint(mx, my):
                        active_slider = key
                        break

            elif event.type == pygame.MOUSEBUTTONUP:
                active_slider = None

            elif event.type == pygame.MOUSEMOTION and active_slider:
                mx, _ = event.pos
                t = np.clip((mx - slider_x) / SLIDER_W, 0, 1)
                s = app.slider_map.get(active_slider)
                if not s:
                    continue
                min_v = s['min']
                max_v = s['max']
                typ = s.get('type', 'int')
                if typ == 'int':
                    val = int(round(min_v + t * (max_v - min_v)))
                elif typ == 'float':
                    step = s.get('step', 0.1)
                    raw = min_v + t * (max_v - min_v)
                    val = round(raw / step) * step
                elif typ == 'bool':
                    val = 1 if t > 0.5 else 0
                else:
                    val = min_v + t * (max_v - min_v)

                setattr(app, active_slider, val)
                action = s.get('action')
                if action and action.get('target') == 'picam2' and hasattr(app, 'picam2'):
                    with contextlib.suppress(Exception):
                        param = action.get('param', 'ExposureTime')
                        ctrl_val = int(val) if isinstance(val, (int, float)) else val
                        app.picam2.set_controls({
                            "AeEnable": False,
                            param: ctrl_val
                        })
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

        # Normal combined view (camera above, mask below)
        combined = pygame.Surface((CAM_W, CAM_H * 2))
        combined.blit(cam_surf, (0, 0))
        combined.blit(mask_surf, (0, CAM_H))

        # If mask-only mode, present only the mask in fullscreen at a logical 320x240
        if getattr(app, 'mask_only', False):
            try:
                # scale mask to logical 320x240 then scale to screen resolution
                mask_small = pygame.transform.smoothscale(mask_surf, (320, 240))
                screen_w, screen_h = screen.get_size()
                full_scaled = pygame.transform.smoothscale(mask_small, (screen_w, screen_h))
                screen.fill((0, 0, 0))
                screen.blit(full_scaled, (0, 0))
                pygame.display.flip()
                clock.tick(PREVIEW_FPS)
                continue
            except Exception:
                # fallback: just blit the mask_surf scaled to screen
                screen_w, screen_h = screen.get_size()
                full_scaled = pygame.transform.smoothscale(mask_surf, (screen_w, screen_h))
                screen.fill((0, 0, 0))
                screen.blit(full_scaled, (0, 0))
                pygame.display.flip()
                clock.tick(PREVIEW_FPS)
                continue

        try:
            simulated = isinstance(app.picam2, DummyPicamera2)
        except Exception:
            simulated = False

        top_label = "Camera (simulated)" if simulated else "Camera"
        bottom_label = "LCD output"
        font = font_small

        def draw_label(surface, text, x, y, padding=6, alpha=160, fixed_w=None):
            txt = font.render(text, True, (255, 255, 255))
            w, h = txt.get_width(), txt.get_height()
            box_w = (fixed_w if fixed_w is not None else w) + padding * 2
            bg = pygame.Surface((box_w, h + padding * 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, alpha))
            surface.blit(bg, (x - padding, y - padding))
            surface.blit(txt, (x, y))

        margin = 6
        top_x = margin
        top_y = CAM_H - font.get_height() - margin
        max_label_text = "Camera (simulated)"
        label_fixed_w = font.size(max_label_text)[0]
        draw_label(combined, top_label, top_x, top_y, fixed_w=label_fixed_w)

        bot_x = margin
        bot_y = CAM_H * 2 - font.get_height() - margin
        draw_label(combined, bottom_label, bot_x, bot_y, fixed_w=label_fixed_w)

        if len(app._preview_time_hist) > 0:
            avg = sum(app._preview_time_hist) / len(app._preview_time_hist)
        else:
            avg = 1.0 / CAM_FPS
        microsec = avg * 1e3
        fps = int(round(1.0 / avg)) if avg > 0 else 0
        ft = f"{microsec:5.1f}ms ({fps:03d} fps)"

        sample = f"{9999.9:07.1f}us ({999:03d} fps)"
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
        text_x = tx + (fixed_w - txt.get_width())
        combined.blit(txt, (text_x, ty))

        scaled = pygame.transform.smoothscale(combined, (target_w, target_h))
        screen.fill((0, 0, 0))
        screen.blit(scaled, (blit_x, blit_y))

        slider_y0 = blit_y + 20
        spacing = 44
        for i, s in enumerate(app.slider_defs):
            y = slider_y0 + spacing * i
            val = getattr(app, s['key'])
            rect = draw_slider(app, screen, slider_x, y, SLIDER_W, SLIDER_H, val, s['min'], s['max'], s['label'])
            app.slider_rects[s['key']] = rect

        pygame.display.flip()
        clock.tick(PREVIEW_FPS)

    pygame.quit()
