CAM_W, CAM_H = 640, 400
CAM_FPS = 60

LCD_W, LCD_H = 320, 240

PREVIEW_FPS = 300

UI_HEIGHT = 120
WINDOW_W = CAM_W
WINDOW_H = CAM_H * 2 + UI_HEIGHT

# Slider layout
SLIDER_X = 20
SLIDER_W = CAM_W - 40
SLIDER_H = 10

SLIDER_Y_EXPOSURE = CAM_H * 2 + 25
SLIDER_Y_CUTOFF   = CAM_H * 2 + 60
SLIDER_Y_DITHER   = CAM_H * 2 + 95

# Sliders
SLIDERS = [
    {"key": "exposure_us",  "label": "Exposure (us)", "min": 500,  "max": 6000, "step": 1,  "default": 3000, "type": "int",
        "action": {"target": "picam2", "method": "set_controls", "param": "ExposureTime"}},
    {"key": "mask_cutoff",  "label": "Mask Cutoff",   "min": 0,    "max": 255,  "step": 1,  "default": 120,  "type": "int"},
    {"key": "dither_enable","label": "Dithering",     "min": 0,    "max": 1,    "step": 1,  "default": 0,    "type": "bool"},
    {"key": "denoise_h",    "label": "Denoise",     "min": 0.0,  "max": 30.0, "step": 0.1,"default": 0,  "type": "float",
        "action": {"target": "mask", "method": "fastNlMeansDenoising", "arg": "h"}},
    {"key": "expand_px",    "label": "Edge Grow",     "min": 0,    "max": 24,   "step": 1,  "default": 0,    "type": "int",
        "action": {"target": "mask", "method": "erode", "arg": "kernel_radius"}},
    {"key": "post_blur",    "label": "Post Blur",     "min": 0.0,  "max": 25.0, "step": 0.1,"default": 0.0,  "type": "float",
        "action": {"target": "mask", "method": "gaussian_blur", "arg": "radius"}},
]
